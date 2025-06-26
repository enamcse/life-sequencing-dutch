import argparse
import logging
import os
import pickle
import re
import sys
import shutil

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
import pandas as pd
import torch
from pytorch_lightning import Trainer

from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
import pop2vec.llm.src.transformer
from pop2vec.llm.src.new_code.load_data import CustomInMemoryDataset
from pop2vec.llm.src.new_code.load_data import CustomIterableDataset
from pop2vec.llm.src.new_code.utils import read_json
from pop2vec.llm.src.new_code.utils import read_hparams
from pop2vec.llm.src.transformer.models import TransformerEncoder

import faiss, time, json
from annoy import AnnoyIndex

PRECISION = "32-true"

logging.basicConfig(
  format="%(asctime)s %(name)s %(levelname)s: %(message)s",
  datefmt="%Y-%m-%d %H:%M:%S",
  level=logging.INFO
)
logger = logging.getLogger(__name__)


def get_callbacks(ckpoint_dir):
  os.makedirs(ckpoint_dir, exist_ok=True)
  callbacks = [
    ModelCheckpoint(
      dirpath=ckpoint_dir,
      filename="model-{epoch:02d}-{step}-{val_loss_track:.2f}",
      monitor="val_loss_track",
      save_top_k=2,
      save_last=False,
      mode='min',
      save_weights_only=False,
      verbose=True,
    )
  ]
  return callbacks

def get_vocab_size(path):
    return len(pd.read_csv(path))

# Helper: Load and update hyperparameters.
def load_hparams(cfg, hparams=None):
    hparams_path = cfg["HPARAMS_PATH"] if hparams is None else hparams
    hparams = read_hparams(hparams_path)
    hparams['vocab_size'] = get_vocab_size(cfg['VOCAB_PATH'])
    hparams.update(cfg)
    return hparams

# Helper: Create training and validation dataloaders.
def get_dataloaders(mlm_path, num_val_items, batch_size):
    val_dataset = CustomInMemoryDataset(
        mlm_path,
        validation=True,
        num_val_items=num_val_items
    )
    train_dataset = CustomInMemoryDataset(
        mlm_path,
        validation=False,
        num_val_items=num_val_items
    )
    num_workers = max(len(os.sched_getaffinity(0)) - 2, 1)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )
    return train_dataloader, val_dataloader


# Helper: Determine the DDP strategy.
def get_ddp_strategy():
    if DDP_STRATEGY == "auto":
        return "auto"
    elif DDP_STRATEGY == "ddp":
        return DDPStrategy()
    elif DDP_STRATEGY == "ddp_mpi":
        return DDPStrategy(process_group_backend="mpi")
    elif DDP_STRATEGY == "gloo":
        return DDPStrategy(process_group_backend="gloo")
    else:
        raise ValueError(f"Unsupported DDP_STRATEGY: {DDP_STRATEGY}")

# Helper: Extract embeddings from the model.
def extract_embeddings(model, dataloader):
    model.eval()
    all_embeddings = []

    with torch.no_grad():
        for batch in dataloader:
            batch_on_device = {
                k: v.to(model.device)
                for k, v in batch.items()
                if isinstance(v, torch.Tensor)
            }
            outputs = model(batch_on_device)
            emb = outputs[0][:, 0, :]  # CLS token
            all_embeddings.append(emb.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)

# Helper: Run Approximate Nearest Neighbor (ANN) benchmarks using Annoy.
def run_annoy(embeddings, sizes, n_trees=10):
    dim = embeddings.shape[1]
    times = {}
    for n in sizes:
        idx = AnnoyIndex(dim, 'euclidean')
        for i, vec in enumerate(embeddings[:n]):
            idx.add_item(i, vec.tolist())
        t0 = time.time()
        idx.build(n_trees)
        times[n] = time.time() - t0
    return times

# Helper: Run FAISS benchmarks on CPU.
def run_faiss_cpu(embeddings, sizes):
    times = {}
    dim = embeddings.shape[1]
    for n in sizes:
        idx = faiss.IndexFlatL2(dim)
        t0 = time.time()
        idx.add(embeddings[:n].astype('float32'))
        times[n] = time.time() - t0
    return times

# Helper: Run FAISS benchmarks on GPU.
def run_faiss_gpu(embeddings, sizes):
    times = {}
    res = faiss.StandardGpuResources()
    dim = embeddings.shape[1]
    for n in sizes:
        cpu_idx = faiss.IndexFlatL2(dim)
        gpu_idx = faiss.index_cpu_to_gpu(res, 0, cpu_idx)
        t0 = time.time()
        gpu_idx.add(embeddings[:n].astype('float32'))
        times[n] = time.time() - t0
    return times

# Main training function.
def pretrain(cfg, batch_size=None, hparams=None):
    ckpt_dir = cfg["CHECKPOINT_DIR"]
    mlm_path = cfg["MLM_PATH"]
    out_dir = cfg["OUTPUT_DIR"]

    # Load hyperparameters.
    hparams = load_hparams(cfg, hparams)

    # Determine batch size and validation interval.
    num_val_items = cfg.get("NUM_VAL_ITEMS", 100000)
    batch_size = hparams['batch_size'] if batch_size is None else batch_size
    val_check_interval = 0.5
    hparams['VAL_CHECK_INTERVAL'] = val_check_interval

    # Create dataloaders.
    train_dataloader, val_dataloader = get_dataloaders(mlm_path, num_val_items, batch_size)
    hparams['steps_per_epoch'] = len(train_dataloader)

    # Set up CSV logger.
    resume_ckpt = cfg.get("RESUME_FROM_CHECKPOINT", None)
    csv_logger = CSVLogger(save_dir=ckpt_dir)

    # Create callbacks.
    callbacks = get_callbacks(ckpt_dir)

    # Decide on the distributed strategy.
    strategy = get_ddp_strategy()

    # Initialize model. The Trainer will load checkpoint state if provided.
    model = TransformerEncoder(hparams)

    # Create Trainer instance. The resume_from_checkpoint argument ensures that
    # model state, optimizer, scheduler (e.g., OneCycleLR), global step, and epoch are resumed.
    trainer = Trainer(
        strategy=strategy,
        default_root_dir=ckpt_dir,
        callbacks=callbacks,
        max_epochs=hparams['epochs'],
        val_check_interval=val_check_interval,
        accelerator=ACCELERATOR,
        devices=N_DEVICES,
        logger=csv_logger,
        precision=PRECISION,
        log_every_n_steps=1000,
    )

    logger.info("Starting Trainer.fit(...)")
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=resume_ckpt)

    logger.info("Training complete. Now extracting embeddings...")
    embeddings = extract_embeddings(model, train_dataloader)
    total = embeddings.shape[0]
    sizes = [10, 100, 1000, 10_000, min(50_000, total), min(100_000, total), total]

    logger.info("Running benchmarks across sizes: %s", sizes)

    # run each
    annoy_times     = run_annoy(embeddings, sizes)
    faiss_cpu_times = run_faiss_cpu(embeddings, sizes)
    faiss_gpu_times = run_faiss_gpu(embeddings, sizes)

    # build DataFrame
    df = pd.DataFrame({
        'embeddings': sizes,
        'annoy'     : [annoy_times[n]    for n in sizes],
        'faiss_cpu' : [faiss_cpu_times[n] for n in sizes],
        'faiss_gpu' : [faiss_gpu_times[n] for n in sizes],
    })

    # ensure step6 folder exists
    bench_dir = os.path.join(out_dir, "step6")
    os.makedirs(bench_dir, exist_ok=True)

    # save CSV
    csv_path = os.path.join(bench_dir, "benchmark_times.csv")
    df.to_csv(csv_path, index=False)
    logger.info("Saved benchmark CSV to %s", csv_path)

    # plot
    plt.figure()
    for col, label in [('annoy','Annoy'), ('faiss_cpu','FAISS CPU'), ('faiss_gpu','FAISS GPU')]:
        plt.plot(df['embeddings'], df[col], marker='o', label=label)
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('# Embeddings'); plt.ylabel('Build Time (s)')
    plt.title('Index Build Time vs Number of Embeddings')
    plt.legend(); plt.grid(True)
    fig_path = os.path.join(bench_dir, "benchmark_plot.png")
    plt.savefig(fig_path)
    logger.info("Saved benchmark plot to %s", fig_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--accelerator", default="gpu", help="Choose an accelerator that connects a Lightning Trainer to arbitrary hardware (CPUs, GPUs, TPUs, HPUs, MPS, ...)")
    parser.add_argument("--ddpstrategy", default="auto", help="pick ddp strategy (auto,gloo,mpi,...)")
    parser.add_argument("--devices", default=1, help="Number of devices")
    parser.add_argument("--batch", default=None, type=int, help="Batch size to use. If None, uses `batch` size specified in the config file")
    parser.add_argument("--hparams", default=None, type=str, help="Path to hyperparameters file. If `None`, uses file specified in the config file")
    parser.add_argument("--config", required=True, help=".json config",type=str)
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    ACCELERATOR=args.accelerator
    N_DEVICES=args.devices
    DDP_STRATEGY=args.ddpstrategy # strategy for pl.Trainer
    BATCH_SIZE=args.batch
    HPARAMS=args.hparams
    CFG_PATH=args.config

    assert DDP_STRATEGY in ["auto", "ddp_mpi", "ddp", "gloo"]

    torch.set_float32_matmul_precision("medium")

    logger.info(f"config path = {CFG_PATH}")
    cfg = read_json(CFG_PATH)
    pretrain(cfg, batch_size=BATCH_SIZE, hparams=HPARAMS)



