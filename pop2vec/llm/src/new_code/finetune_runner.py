"""
finetune_runner.py  - -  Multi‑GPU ready
======================================
Launch fine‑tuning runs (one per target column) and log results
to exactly the same CSV schema as 'train_simple.py'.

New keys understood in the JSON config (all optional):
    devices                : int   (default 1)
    accelerator            : str   (default "gpu")
    ddpstrategy            : str   (default "auto" -> same as Lightning)
    accumulate_grad_batches: int   (default 1)
    gradient_clip_val      : float (default 1)
    training_precision     : str   (default "32-true")
    val_check_interval     : float (default 1.0)
"""
from __future__ import annotations
import json, os, sys, logging, csv
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

# ——— import the existing fine‑tune helpers ————————————————
import pop2vec.llm.src.new_code.finetune_new as finetune
from pop2vec.llm.src.new_code.utils import read_hparams
from pop2vec.llm.src.new_code.load_data import FineTuneLazyDataset
from pop2vec.llm.src.new_code.finetune_model import TransformerFT
from pop2vec.evaluation.prediction_settings.train_simple import _en_weights
from pytorch_lightning.strategies import DDPStrategy

import copy
import torch.distributed as distributed
import gc

import os
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_squared_error,
    r2_score,
    roc_auc_score, 
)

# ─────────────────── Config helpers ───────────────────────────────────
DEFAULTS = {
    "EARLY_STOP_PATIENCE": 3,
    "MAX_EPOCHS": 2000,
    "LR": 1e-6,
    "BATCH_SIZE": 32,
    "balance_dataset": False,
    "test_only": False,
    #  NEW multi‑GPU / trainer defaults 
    "devices": 4,
    "freeze_positions": False,
    "pooled": True,
    "class-balanced-loss": False,
    "accelerator": "gpu",
    "ddpstrategy": "auto",
    "accumulate_grad_batches": 1,
    "gradient_clip_val": 1,
    "training_precision": "bf16",
    "val_check_interval": 1.0,
    "weight_decay_enc": 1e-2,
    "layer_lr_decay": 0.95,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-6,
    "optimizer_type": "adamw",
    "lr_scheduler": "onecycle",
    "binary_threshold": 0.5,    
}

ALWAYS_REQUIRED = [
    "sequence_encoded",
    "result_dir",
    "task_file",       
    "target_column",
]
REQUIRED_TRAIN = ["train_path", "val_path", "model_save_dir", "pretrained_model_path", "pretrained_model_hparams", "model_name",]

REQUIRED_TEST =  ["test_path", "load_model_path",]

def _with_defaults(cfg: Dict) -> Dict:
    out = cfg.copy()
    for k, v in DEFAULTS.items():
        out.setdefault(k, v)
    if out.get("result_path", None) is None:
        out["result_path"] = str(Path(out["result_dir"], f"{out['task_file']}_{out['target_column']}.csv"))
    return out

def _integrity_check(cfg: Dict):
    missing = [k for k in ALWAYS_REQUIRED if k not in cfg]
    if cfg.get("test_only"):
        missing += [k for k in REQUIRED_TEST if k not in cfg]
    else:
        missing += [k for k in REQUIRED_TRAIN if k not in cfg]
    
    if missing:
        raise ValueError("Missing required keys: " + ", ".join(missing))

def _fmt(metrics: Dict, key: str):
    v = metrics.get(key, None)
    if v is None:
        return ""
    if isinstance(v, torch.Tensor):
        v = v.item()
    return f"{v:.4f}"

def _write_row(result_path: str, header: List[str], row: List):
    need_hdr = not os.path.exists(result_path) or os.path.getsize(result_path) == 0
    with open(result_path, "a", newline="") as f:   
        w = csv.writer(f)
        if need_hdr:
            w.writerow(header)
        w.writerow(row)

def _read_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)

# -----------------------------------------------------------------------------
# Metrics ----------------------------------------------------------------------
# -----------------------------------------------------------------------------

def _best_f1_threshold_torch(p: torch.Tensor, y: torch.Tensor):
        """Return (thr*, f1*) using the exact sweep."""
        # sort descending by probability
        p_sorted, idx = torch.sort(p, descending=True)
        y_sorted = y[idx]

        tp = torch.cumsum(y_sorted == 1, 0)
        fp = torch.cumsum(y_sorted == 0, 0)
        fn = tp[-1] - tp

        precision = tp / (tp + fp + 1e-12)
        recall    = tp / (tp + fn + 1e-12)
        f1        = 2 * precision * recall / (precision + recall + 1e-12)

        best_idx = torch.argmax(f1)
        return p_sorted[best_idx], f1[best_idx]

# ─────────────────── Lightning helpers ────────────────────────────────
def _build_hparams(cfg: Dict, target_col: str,
                   target_type: str, num_outputs: int) -> Dict:
    ft_hp = {
        "finetune_checkpoint_dir": str(Path(cfg["model_save_dir"]) / target_col),
        # "sequence_encoded": cfg["sequence_encoded"],
        "train_label_file": cfg["train_path"],
        "val_label_file":   cfg["val_path"],
        # "pretrained_model_path":     cfg["pretrained_model_path"],
        # "pretrained_model_hparams":  cfg["pretrained_model_hparams"],
        "batch_size":  cfg["BATCH_SIZE"],
        "epochs":      cfg["MAX_EPOCHS"],
        "learning_rate": cfg["LR"],
        "num_targets": num_outputs,
        "target_col":  target_col,
        "oversample":  cfg["balance_dataset"],
        "val_split":   0.0,   # external val file
        # propagate trainer‑level knobs so they survive enc_hp merge
        # "accumulate_grad_batches": cfg["accumulate_grad_batches"],
        # "gradient_clip_val":       cfg["gradient_clip_val"],
        # "training_precision":      cfg["training_precision"],
        # "val_check_interval":      cfg["val_check_interval"],
        "task_type": target_type,
    }
    for k, v in cfg.items():
        ft_hp[k] = v
    enc_hp = read_hparams(cfg["pretrained_model_hparams"])
    enc_hp.update(ft_hp)
    return enc_hp

def _monitor_and_mode(task_type: str) -> Tuple[str, str]:
    if task_type == "numeric":
        return "val_r2_epoch", "max"
    return "val_auc_epoch", "max"      # binary + categorical

def _get_ddp_strategy(name: str):
    if name == "auto":
        return "auto"
    if name == "ddp":
        return DDPStrategy()
    if name == "ddp_mpi":
        return DDPStrategy(process_group_backend="mpi")
    if name == "gloo":
        return DDPStrategy(process_group_backend="gloo")
    raise ValueError(f"Unknown ddpstrategy '{name}'")

def transform_categorical_labels(loader, num_outputs):
    labels_tensor = loader.dataset.labels_tensor
    if bool(1 <= torch.min(labels_tensor)) and bool(torch.max(labels_tensor) <= num_outputs):
        labels_tensor -= 1
        loader.dataset.labels_tensor = labels_tensor
        return loader
    else:
        raise ValueError(
            f"labels must be between 1 and num_ouputs = {num_outputs}, found min = {np.min(labels_tensor)} max = {np.max(labels_tensor)}"
        )

def _train_one_target(cfg, target_col, target_type, num_outputs):
    hp = _build_hparams(cfg, target_col, target_type, num_outputs)
    monitor, mode = _monitor_and_mode(target_type)

    os.makedirs(hp["finetune_checkpoint_dir"], exist_ok=True)

    # — dataloaders —
    train_loader, val_loader = finetune.get_dataloaders(hp)
    if target_type == 'categorical':
        train_loader = transform_categorical_labels(train_loader, num_outputs)
        val_loader = transform_categorical_labels(val_loader, num_outputs)
    if cfg['class-balanced-loss']:
        _labels = train_loader.dataset._labels
        counts = np.bincount(
            np.array(_labels, dtype='long'), 
            minlength=num_outputs
        ).tolist()
        weights  = _en_weights(counts)
        logging.info(f"counts = {counts}")
        logging.info(f"weights = {weights}")
        hp['loss_weights'] = weights

    if target_type == 'numeric':
        hp['sigma'] = train_loader.dataset.labels_tensor.std(unbiased=False)
        hp['mu'] = train_loader.dataset.labels_tensor.mean()


    # keep lr‑scheduler logic in FT model happy
    acc_grad = hp["accumulate_grad_batches"]
    hp["steps_per_epoch"] = int(len(train_loader) /
                                (cfg["devices"] * acc_grad)) + 2

    model = TransformerFT(hp)

    # — callbacks —
    ckpt_cb = ModelCheckpoint(
        dirpath=hp["finetune_checkpoint_dir"],
        filename=f"finetune-{{epoch:02d}}-{{step}}-{{{monitor}:.2f}}",
        monitor=monitor, mode=mode, save_top_k=1, verbose=True,
    )
    early_cb = EarlyStopping(
        monitor=monitor, mode=mode,
        patience=cfg["EARLY_STOP_PATIENCE"], verbose=True, min_delta=0.001
    )
    logger = CSVLogger(save_dir=hp["finetune_checkpoint_dir"])

    strategy = _get_ddp_strategy(cfg["ddpstrategy"])

    trainer = Trainer(
        strategy=strategy,
        default_root_dir=hp["finetune_checkpoint_dir"],
        accelerator=cfg["accelerator"],
        devices=cfg["devices"],
        max_epochs=hp["epochs"],
        callbacks=[early_cb, ckpt_cb],
        logger=logger,
        precision=hp["training_precision"],
        log_every_n_steps=250,
        gradient_clip_val=hp["gradient_clip_val"],
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=hp["accumulate_grad_batches"],
        val_check_interval=hp["val_check_interval"],
    )

    trainer.fit(model, train_loader, val_loader)

    best_ckpt = trainer.checkpoint_callback.best_model_path

    if target_type == 'binary' and best_ckpt:
        # Build val loader (same as before)
        val_ds = FineTuneLazyDataset(
            h5_file_path=cfg['sequence_encoded'],
            train_file_path=cfg['val_path'],
            target_col=target_col,
            phase="validation",
            return_sequence_id=True,
            primary_key=hp.get('PRIMARY_KEY', 'RINPERSOON'),
        )
        nw = 0  # tiny set; avoid spawning workers that can slow/hang on HDF5
        val_dl = torch.utils.data.DataLoader(val_ds, batch_size=cfg["BATCH_SIZE"], shuffle=False, num_workers=nw)

        # In recent version, lets name it v1, the following line is added here
        if trainer.is_global_zero: 
            single = Trainer(accelerator="cpu", devices=1, logger=False, enable_progress_bar=False
            ) # Added in v1
            # Load best checkpoint (read-only)
            val_model = TransformerFT.load_from_checkpoint(
                best_ckpt, task_type=target_type, pretrained_model_path=cfg['pretrained_model_path'], strict=False
            )
            val_model.eval()
            

            # Distributed predict on ALL ranks
            outs = single.predict(val_model, val_dl, return_predictions=True) # Changed in v1
            # val_outputs = trainer.predict(val_model, val_dl, return_predictions=True)
            logging.info(f"outs length = {len(outs)}, outs[0] keys = {outs[0].keys()}")
            # Local tensors
            key = cfg.get("PRIMARY_KEY", "RINPERSOON")
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # local_logits = torch.cat([out["preds"] for out in val_outputs]).to(device)
            # local_ids    = torch.cat([out[key] for out in val_outputs]).to(device)
            local_logits = torch.cat([out["preds"] for out in outs]).float()
            local_ids    = torch.cat([out[key] for out in outs]).cpu().numpy()

            # Gather across ranks: commented out in v1
            # g_logits = trainer.strategy.all_gather(local_logits)
            # g_ids    = trainer.strategy.all_gather(local_ids)

            # Flatten gathered tensors
            # if g_logits.dim() == 2:  # (world, N)
            #     logits_all = g_logits.flatten(0, 1).cpu()
            #     ids_all    = g_ids.flatten(0, 1).cpu().numpy()
            # else:
            #     logits_all = g_logits.cpu()
            #     ids_all    = g_ids.cpu().numpy()

            # Convert logits -> probabilities
            # if logits_all.ndim == 2 and logits_all.size(1) == 2:
            #     prob_all = torch.softmax(logits_all, dim=1)[:, 1]
            # else:
            #     prob_all = torch.sigmoid(logits_all.squeeze())

            if local_logits.ndim == 2 and local_logits.size(1) == 2:
                prob_all = torch.softmax(local_logits, dim=1)[:, 1]
            else:
                prob_all = torch.sigmoid(local_logits.squeeze())

            # Only global zero does I/O + threshold writeback
            # if trainer.is_global_zero: # moved up in v1
            val_ids_df = _read_any(cfg['val_path'])
            # y_series = (
            #     pd.DataFrame({key: ids_all})
            #     .merge(val_ids_df[[key, target_col]], on=key, how="left", validate="m:1")[target_col]
            # )
            y_series = (
                pd.DataFrame({key: local_ids})
                .merge(val_ids_df[[key, target_col]], on=key, how="left", validate="m:1")[target_col]
            )
            y_val = torch.from_numpy(y_series.to_numpy().astype(int))

            best_thr_t, _ = _best_f1_threshold_torch(prob_all, y_val)

            raw = torch.load(best_ckpt, map_location='cpu')
            raw["best_thr"] = best_thr_t.detach().cpu()
            torch.save(raw, best_ckpt)
            logging.info(f"Saved best_thr = {float(best_thr_t):.6f} to {best_ckpt}")

        # Make sure all ranks see the updated file before validate
        trainer.strategy.barrier()

    best_metrics = trainer.validate(
        model=model, dataloaders=val_loader,
        ckpt_path="best", verbose=False,
    )[0]

    trainer.strategy.teardown()
    if distributed.is_initialized():
        distributed.destroy_process_group()
    torch.cuda.empty_cache()
    gc.collect()
    
    return best_metrics, ckpt_cb.best_model_path


def _is_dist():
    return distributed.is_available() and distributed.is_initialized()

def _rank():
    return distributed.get_rank() if _is_dist() else 0

def _world():
    return distributed.get_world_size() if _is_dist() else 1

def _ddp_log(msg: str):
    print(f"[DDP r{_rank()}/{_world()}] {msg}")

def _shape(x):
    try:
        return tuple(x.shape)
    except Exception:
        try:
            return (len(x),)
        except Exception:
            return ("?",)

def _logits_to_pos_proba(preds, target_type: str, num_out: int):
    """
    Convert model outputs to P(class=1) as numpy array of shape (N,).
    - For binary BCE-with-logits: preds shape (N,) or (N,1) -> sigmoid
    - For 2-class softmax logits: preds shape (N,2) -> softmax[:,1]
    - For multiclass K>2: preds shape (N,K) -> softmax; caller picks columns
    """
    t = torch.as_tensor(preds)
    if target_type == "binary":
        # handle (N,), (N,1), or (N,2)
        if t.ndim == 1:
            # BCE-with-logits common case
            return torch.sigmoid(t).cpu().numpy()
        elif t.ndim == 2:
            if t.size(1) == 1:
                # single-logit head
                return torch.sigmoid(t[:, 0]).cpu().numpy()
            elif t.size(1) == 2:
                # two-logit softmax head
                return torch.softmax(t, dim=1)[:, 1].cpu().numpy()
            else:
                raise ValueError(f"Binary expected 1 or 2 logits, got shape {tuple(t.shape)}")
        else:
            raise ValueError(f"Unexpected preds ndim={t.ndim} for binary")
    else:
        # multiclass/regression pass-through (adjust if you need per-class probs)
        if t.ndim == 2 and num_out > 1:
            return torch.softmax(t, dim=1).cpu().numpy()  # shape (N,K)
        return t.cpu().numpy()

def _run_test(c: Dict, tgt: str, ttype: str, k_out: int):
    _ddp_log("_run_test start")

    # Enable CUDA synchronous execution for better debugging
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # Initialize CUDA context properly
    if torch.cuda.is_available():
        torch.cuda.init()
        torch.cuda.empty_cache()
        # Set matmul precision for stability
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        print(f"CUDA initialized: device count = {torch.cuda.device_count()}")
        print("CUDA_LAUNCH_BLOCKING enabled for debugging")

    ds = FineTuneLazyDataset(
        h5_file_path=c['sequence_encoded'],
        train_file_path=c['test_path'],
        phase="test",
        return_sequence_id=True,
        primary_key=c.get('PRIMARY_KEY', 'RINPERSOON'),
    )
    
    # DEBUG: Print the data sources being used
    logging.info("=== DATA SOURCE DEBUG INFO ===")
    logging.info(f"H5 sequence file: {c['sequence_encoded']}")
    logging.info(f"Test labels file: {c['test_path']}")
    logging.info(f"Model checkpoint: {c['load_model_path']}")
    
    # Check if we can get vocabulary info from the H5 file itself
    try:
        import h5py
        with h5py.File(c['sequence_encoded'], 'r') as f:
            if 'vocab_size' in f.attrs:
                h5_vocab_size = f.attrs['vocab_size']
                logging.info(f"H5 file vocabulary size: {h5_vocab_size}")
            else:
                logging.info("H5 file does not contain vocab_size attribute")
                
            # Check what datasets are in the H5 file
            logging.info(f"H5 file datasets: {list(f.keys())}")
            if 'sequences' in f:
                seq_shape = f['sequences'].shape
                logging.info(f"H5 sequences shape: {seq_shape}")
    except Exception as e:
        logging.warning(f"Could not read H5 file info: {e}")
    logging.info("=== END DATA SOURCE DEBUG INFO ===")
    nw = len(os.sched_getaffinity(0)) - 1
    logging.info(f"num_workers = {nw}")
    dl = torch.utils.data.DataLoader(
        ds, batch_size=c["BATCH_SIZE"], shuffle=False, num_workers=nw
    )

    try:
        _ddp_log(f"dl.sampler={type(dl.sampler).__name__}")
    except Exception:
        pass

    model = TransformerFT.load_from_checkpoint(
        c["load_model_path"], task_type=ttype, pretrained_model_path='RANDOM'
    )
    
    # DEBUG: Print model source and vocabulary info
    logging.info("=== MODEL SOURCE DEBUG INFO ===")
    logging.info(f"Model loaded from checkpoint: {c['load_model_path']}")
    
    # Try to extract hyperparameters that show original vocabulary size
    if hasattr(model, 'hparams'):
        original_vocab = getattr(model.hparams, 'vocab_size', 'unknown')
        logging.info(f"Model hparams vocab_size: {original_vocab}")
        
        # Check if there are other relevant hparams
        for key in ['pretrained_model_path', 'pretrained_model_hparams']:
            if hasattr(model.hparams, key):
                logging.info(f"Model hparams {key}: {getattr(model.hparams, key)}")
    
    # Check actual embedding layer size
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'embedding'):
        actual_vocab = model.transformer.embedding.token.num_embeddings
        logging.info(f"Actual embedding layer vocab_size: {actual_vocab}")
    elif hasattr(model, 'encoder') and hasattr(model.encoder, 'embeddings'):
        actual_vocab = model.encoder.embeddings.token.num_embeddings
        logging.info(f"Actual embedding layer vocab_size: {actual_vocab}")
        
    logging.info("=== END MODEL SOURCE DEBUG INFO ===")

    # DEBUGGING: Check vocabulary sizes and token ID ranges
    logging.info("=== VOCABULARY DEBUG INFO ===")
    vocab_size = None
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'embedding'):
        embedding_layer = model.transformer.embedding.token
        vocab_size = embedding_layer.num_embeddings
        logging.info(f"Model vocabulary size: {vocab_size}")
    elif hasattr(model, 'encoder') and hasattr(model.encoder, 'embeddings'):
        embedding_layer = model.encoder.embeddings.token
        vocab_size = embedding_layer.num_embeddings
        logging.info(f"Model vocabulary size: {vocab_size}")
    else:
        logging.warning("Could not access embedding layer for vocabulary size check")
    
    if vocab_size is not None:
        # Check a few batches for token ID ranges
        logging.info("Checking token ID ranges in test data...")
        max_token_id = -1
        min_token_id = float('inf')
        
        for i, batch in enumerate(dl):
            if i >= 3:  # Check only first 3 batches
                break
            input_ids = batch['input_ids']  # Shape: (batch, 4, seq_len)
            token_ids = input_ids[:, 0, :]  # First dimension is tokens
            
            batch_max = torch.max(token_ids).item()
            batch_min = torch.min(token_ids[token_ids > 0]).item()  # Exclude padding (0)
            
            max_token_id = max(max_token_id, batch_max)
            min_token_id = min(min_token_id, batch_min)
            
            logging.info(f"Batch {i}: token_ids range [{batch_min}, {batch_max}]")
        
        logging.info(f"Overall token ID range in test data: [{min_token_id}, {max_token_id}]")
        
        if max_token_id >= vocab_size:
            logging.error(f"PROBLEM FOUND: Max token ID ({max_token_id}) >= vocabulary size ({vocab_size})")
            logging.error(f"This will cause index out of bounds errors!")
            
            # Try to identify which tokens are problematic
            problematic_ids = []
            for i, batch in enumerate(dl):
                if i >= 10:  # Check first 10 batches
                    break
                input_ids = batch['input_ids']
                token_ids = input_ids[:, 0, :]
                invalid_mask = token_ids >= vocab_size
                if invalid_mask.any():
                    invalid_tokens = token_ids[invalid_mask].unique()
                    problematic_ids.extend(invalid_tokens.tolist())
            
            problematic_ids = sorted(set(problematic_ids))
            logging.error(f"Problematic token IDs (>= vocab_size): {problematic_ids[:20]}...")  # Show first 20
            
            return None  # Exit early to avoid crash
        else:
            logging.info("✅ Token ID ranges look good!")
    
    logging.info("=== END VOCABULARY DEBUG INFO ===")

    # Clean up hyperparameters to avoid TensorBoard issues
    if hasattr(model, 'hparams'):
        # Remove potentially problematic hyperparameters that TensorBoard can't handle
        problematic_keys = []
        for key, value in model.hparams.items():
            try:
                # Test if the value can be converted to a format TensorBoard accepts
                if isinstance(value, (dict, list, tuple)) and len(str(value)) > 1000:
                    problematic_keys.append(key)
                elif hasattr(value, 'shape') and len(value.shape) > 2:  # Multi-dimensional tensors
                    problematic_keys.append(key) 
                elif isinstance(value, torch.Tensor) and value.numel() > 100:  # Large tensors
                    problematic_keys.append(key)
            except Exception:
                problematic_keys.append(key)
        
        # Remove problematic hyperparameters
        for key in problematic_keys:
            if key in model.hparams:
                logging.info(f"Removing problematic hyperparameter for logging: {key}")
                delattr(model.hparams, key)

    thr_from_ckpt = None

    if hasattr(model, "best_thr"):
        try:
            thr_from_ckpt = float(model.best_thr)
            logging.info(f">>>>>> Loaded best_thr = {thr_from_ckpt:.6} from checkpoint")
        except Exception:
            thr_from_ckpt = None

    if (thr_from_ckpt is None) or np.isnan(thr_from_ckpt):
        try:
            raw = torch.load(c["load_model_path"], map_location='cpu')
            sd = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
            if "best_thr" in sd:
                thr_from_ckpt = float(sd["best_thr"])
                logging.info(f">>>>>> Loaded best_thr = {thr_from_ckpt:.6} from state_dict")
        except Exception:
            thr_from_ckpt = None
    
    if (thr_from_ckpt is None) or np.isnan(thr_from_ckpt):
        logging.info("Model has no saved threshold; using config")
        thr_from_ckpt = float(c.get("binary_threshold", 0.5))
    
    logging.info(f"Using threshold = {thr_from_ckpt:.6}")

    strategy = _get_ddp_strategy(c["ddpstrategy"])

    # Create a custom dataset wrapper to fix token ID issues
    if vocab_size is not None:
        logging.info(f"Wrapping dataset with vocabulary size limit: {vocab_size}")
        
        class SafeDataset(torch.utils.data.Dataset):
            def __init__(self, original_dataset, vocab_size):
                self.original_dataset = original_dataset
                self.vocab_size = vocab_size
                self.replacements_made = 0
                
            def __len__(self):
                return len(self.original_dataset)
            
            def __getitem__(self, idx):
                sample = self.original_dataset[idx]
                
                # Fix token IDs that are out of bounds
                if 'input_ids' in sample:
                    input_ids = sample['input_ids'].clone()
                    
                    # Check first dimension (tokens) 
                    token_ids = input_ids[0, :]
                    out_of_bounds_mask = token_ids >= self.vocab_size
                    
                    if out_of_bounds_mask.any():
                        # Log first few replacements for debugging
                        if self.replacements_made < 5:
                            invalid_tokens = token_ids[out_of_bounds_mask].unique()
                            logging.warning(f"Sample {idx}: Replacing out-of-bounds tokens {invalid_tokens.tolist()} with PAD (0)")
                        
                        # Replace out-of-bounds tokens with [PAD] (ID 0)
                        input_ids[0, out_of_bounds_mask] = 0  # Use PAD token
                        sample['input_ids'] = input_ids
                        
                        # Update padding mask to mark these as padding
                        if 'padding_mask' in sample:
                            padding_mask = sample['padding_mask'].clone()
                            padding_mask[out_of_bounds_mask] = 0
                            sample['padding_mask'] = padding_mask
                            
                        self.replacements_made += 1
                
                return sample
        
        # Wrap the dataset
        safe_ds = SafeDataset(ds, vocab_size)
        dl = torch.utils.data.DataLoader(
            safe_ds, batch_size=c["BATCH_SIZE"], shuffle=False, num_workers=nw
        )
        logging.info("✅ Dataset wrapped with vocabulary safety layer")
    else:
        logging.warning("❌ Could not determine vocabulary size - proceeding without safety wrapper")

    trainer = Trainer(
        strategy=strategy,
        accelerator=c["accelerator"],
        devices=c["devices"],
        logger=False,# Re-enable logging now that we've cleaned the hyperparameters
    )

    _ddp_log("after load_from_checkpoint + trainer init")

    # Ensure model is properly on GPU and synchronized
    if torch.cuda.is_available():
        model = model.cuda()
        torch.cuda.synchronize()
        print(f"Model moved to GPU and synchronized")
        
        # Warm up CUDA with a small forward pass
        try:
            with torch.no_grad():
                dummy_batch = next(iter(dl))
                print("Testing model with dummy input...")
                # Ensure all tensors are on the same device (GPU) - be more explicit
                batch_dict = {}
                for k, v in dummy_batch.items():
                    if k in ['input_ids', 'padding_mask']:
                        if torch.is_tensor(v):
                            batch_dict[k] = v.cuda()
                        else:
                            batch_dict[k] = v
                
                print(f"Batch dict devices: {[(k, v.device if torch.is_tensor(v) else type(v)) for k, v in batch_dict.items()]}")
                print(f"Model device: {next(model.parameters()).device}")
                
                _ = model(batch_dict)
                print("✅ Dummy forward pass successful")
                torch.cuda.synchronize()
        except Exception as e:
            print(f"❌ Dummy forward pass failed: {e}")
            # Print more debugging info
            print(f"Model parameters device: {next(model.parameters()).device}")
            if 'batch_dict' in locals():
                for k, v in batch_dict.items():
                    if torch.is_tensor(v):
                        print(f"  {k}: shape={v.shape}, device={v.device}, dtype={v.dtype}")
                    else:
                        print(f"  {k}: type={type(v)}")
            # Continue anyway, but this warns us about issues

    outputs = trainer.predict(model, dl, return_predictions=True)

    _ddp_log(f"predict done; batches={len(outputs)}")
    if len(outputs) > 0 and isinstance(outputs[0], dict):
        keys0 = list(outputs[0].keys())
        _ddp_log(f"first output keys={keys0}")
        if "preds" in outputs[0]:
            print("First preds shape:", _shape(outputs[0]["preds"]))
        if c.get("PRIMARY_KEY", "RINPERSOON") in outputs[0]:
            print("first ids shape:", _shape(outputs[0][c.get("PRIMARY_KEY", "RINPERSOON")]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # collect local tensors
    local_preds = torch.cat([out["preds"] for out in outputs]).to(device)
    local_ids   = torch.cat([out[c.get("PRIMARY_KEY", "RINPERSOON")] for out in outputs]).to(device)
    _ddp_log(f"local_preds shape={tuple(local_preds.shape)}, local_ids shape={tuple(local_ids.shape)}")
    logging.info(f"local_preds.shape = {local_preds.shape}")

    # --- NEW: all-gather + flatten while keeping the class dim intact ---
    g_preds = trainer.strategy.all_gather(local_preds)  # torch.Tensor
    g_ids   = trainer.strategy.all_gather(local_ids)
    _ddp_log(f"gathered preds shape={tuple(g_preds.shape)}, ids shape={tuple(g_ids.shape)}")

    # keep last dim (class K) intact
    if g_preds.dim() == 3:           # (world, N, K)
        preds = g_preds.flatten(0, 1).cpu().numpy()   # -> (world*N, K)
    elif g_preds.dim() == 2:         # (N, K)
        preds = g_preds.cpu().numpy()
    elif g_preds.dim() == 1:         # (N,)
        preds = g_preds.cpu().numpy()
    else:
        raise ValueError(f"unexpected g_preds.dim={g_preds.dim()}")

    if g_ids.dim() == 2:             # (world, N)
        ids = g_ids.flatten(0, 1).cpu().numpy()      # -> (world*N,)
    elif g_ids.dim() == 1:           # (N,)
        ids = g_ids.cpu().numpy()
    else:
        raise ValueError(f"unexpected g_ids.dim={g_ids.dim()}")
    _ddp_log(f"gathered preds shape={tuple(local_preds.shape)}, local_ids shape={local_ids.shape}")

    # only rank 0 writes/returns
    if not trainer.is_global_zero:
        _ddp_log("non-zero rank; skip write")
        return

    test_metrics = None
    
    # --- Common setup for metrics calculation ---
    try:
        key = c.get("PRIMARY_KEY", "RINPERSOON")
        labels_df = _read_any(c.get("test_path", '/gpfs/ostor/ossc9424/data/PreFer/sample_size_exp/dataset/test/final_leaderboard_outcome.parquet'))

        if key in labels_df.columns and tgt in labels_df.columns:
            lbls_series = (
                pd.DataFrame({key: ids})
                .merge(labels_df[[key, tgt]], on=key, how="left", validate="m:1")[tgt]
            )
            y_true = lbls_series.to_numpy()
            
            mask = ~pd.isna(y_true)
            if mask.any():
                y_true_m = y_true[mask]
                preds_m = preds[mask]

                if ttype == "binary":
                    probs_m = _logits_to_pos_proba(preds_m, target_type="binary", num_out=2)
                    pred_lbls_m = (probs_m >= thr_from_ckpt).astype(int)
                    y_true_m = y_true_m.astype(int)
                    test_metrics = {
                        "acc": accuracy_score(y_true_m, pred_lbls_m),
                        "f1": f1_score(y_true_m, pred_lbls_m, average="binary"),
                        "auc": roc_auc_score(y_true_m, probs_m),
                        "mcc": matthews_corrcoef(y_true_m, pred_lbls_m),
                        "mae": "", "r2": "",
                    }

                elif ttype == "categorical":
                    pred_indices = np.argmax(preds_m, axis=1)
                    y_true_m = y_true_m.astype(int)
                    if np.min(y_true_m) == 1:
                        y_true_m -= 1
                    
                    probas_m = torch.softmax(torch.from_numpy(preds_m), dim=1).numpy()
                    test_metrics = {
                        "acc": accuracy_score(y_true_m, pred_indices),
                        "f1": f1_score(y_true_m, pred_indices, average="macro"),
                        "auc": roc_auc_score(y_true_m, probas_m, average="macro", multi_class="ovr"),
                        "mcc": matthews_corrcoef(y_true_m, pred_indices),
                        "mae": "", "r2": "",
                    }

                elif ttype == "numeric":
                    y_true_m = y_true_m.astype(float)
                    preds_m = preds_m.astype(float).flatten()
                    
                    # --- Un-normalize predictions ---
                    # The model was trained on normalized labels (z-score).
                    # We must un-normalize its predictions before comparing to true labels.
                    mu = model.hparams.get("mu", 0.0)
                    sigma = model.hparams.get("sigma", 1.0)
                    
                    # Convert tensors to floats for numpy operations
                    if hasattr(mu, 'item'):
                        mu = mu.item()
                    if hasattr(sigma, 'item'):
                        sigma = sigma.item()

                    if sigma == 0: sigma = 1.0 # Avoid division by zero
                    
                    preds_unnormalized = (preds_m * sigma) + mu
                    
                    logging.info(f"Un-normalizing numeric predictions with mu={mu:.4f}, sigma={sigma:.4f}")

                    test_metrics = {
                        "acc": "", "f1": "", "auc": "", "mcc": "",
                        "mae": np.mean(np.abs(y_true_m - preds_unnormalized)),
                        "r2": r2_score(y_true_m, preds_unnormalized),
                    }
                    # Also un-normalize the full `preds` array for saving
                    preds = (preds.astype(float).flatten() * sigma) + mu

                if test_metrics:
                    logging.info("Test metrics: " + ", ".join(f"{k}={v:.4f}" for k, v in test_metrics.items() if v != ""))
            else:
                logging.warning("No non-NA labels found; skipping test metrics")
        else:
            logging.warning(f"Key columns not found in labels_df: {key}, {tgt}")
    except Exception as e:
        logging.warning(f"Could not compute test metrics: {e}", exc_info=True)

    # --- Prepare output file ---
    if ttype == "binary":
        probs = _logits_to_pos_proba(preds, target_type="binary", num_out=2)
        lbls  = (probs >= thr_from_ckpt).astype(int)
        if not (len(ids) == len(probs) == len(lbls)):
            raise ValueError(f"Length mismatch: ids={len(ids)} probs={len(probs)} lbls={len(lbls)}")
        arr  = np.c_[ids, probs, lbls]
        hdr  = "RINPERSOON,probability,prediction"
    else:
        if ttype == "categorical":
            preds = np.asarray(preds).argmax(1) + 1  # 1-based labels
        else: # numeric - already un-normalized
            preds = np.asarray(preds)
        
        ids   = np.asarray(ids).reshape(-1)
        preds = preds.reshape(-1)

        if len(ids) != len(preds):
            raise ValueError(f"Length mismatch: ids={len(ids)} preds={len(preds)}")
        arr = np.c_[ids, preds]
        hdr = "RINPERSOON,prediction"

    if c.get("save_predictions", None):
        out = Path(c["result_dir"], f"{c['task_file']}_{tgt}.csv")
        np.savetxt(out, arr, delimiter=",", header=hdr, comments="", fmt="%s")
        _ddp_log(f"wrote {out}")
    else:
        _ddp_log("save_predictions not set; skipping predictions file write")

    return test_metrics

def should_work(path):
  if not Path(path).is_file():
    return True              
  with open(path) as f:
    return sum(1 for _ in f) < 2 


# ─────────────────── main loop ────────────────────────────────────────
def main(cfg_path: str):
    cfg = _with_defaults(json.load(open(cfg_path)))
    _integrity_check(cfg)
    if not should_work(cfg['result_path']):
        logging.info("work was already done. Exiting without doing anything.")
        return
    header = [
        "mode", "task_file", "model_name", "target", "type", "model_path",
        "val_acc", "val_f1", "val_auc", "val_mcc", "val_mae", "val_r2",
        "test_acc", "test_f1", "test_auc", "test_mcc", "test_mae", "test_r2",
        "LR", "BATCH-SIZE"
    ]
    if cfg.get("sample_size", None):
        header += ["sample_size"]

    os.makedirs(cfg["result_dir"], exist_ok=True)

    for tgt_col, (tgt_type, k_out) in cfg["target_column"].items():
        logging.info(f"=== Fine‑tune '{tgt_col}' ({tgt_type}) ===")
        cfg_copy = copy.deepcopy(cfg)
        row = []
        if cfg_copy["test_only"]:
            logging.info(f"--- Test-only evaluation for '{tgt_col}' ({tgt_type}) ---")
            test_metrics = _run_test(cfg_copy, tgt_col, tgt_type, k_out)
            if (test_metrics is not None) and (os.getenv("SLURM_PROCID", "0") == "0"):
                row = [
                    "test", cfg_copy["task_file"], cfg_copy["model_name"], tgt_col, tgt_type,
                    cfg_copy.get("load_model_path", ""),
                    "", "", "", "", "", "",
                    _fmt(test_metrics, "acc"),
                    _fmt(test_metrics, "f1"),
                    _fmt(test_metrics, "auc"),
                    _fmt(test_metrics, "mcc"),
                    _fmt(test_metrics, "mae"),
                    _fmt(test_metrics, "r2"),
                    cfg_copy.get("LR", ""),
                    cfg_copy.get("BATCH_SIZE", ""),
                ]
                
        else:
            val_metrics, model_path = _train_one_target(
                cfg_copy, tgt_col, tgt_type, k_out
            )

            row = [
                "train", cfg_copy["task_file"], cfg_copy["model_name"], tgt_col, tgt_type,
                model_path,
                _fmt(val_metrics, "val_acc_epoch"),
                _fmt(val_metrics, "val_f1_epoch"),
                _fmt(val_metrics, "val_auc_epoch"),
                _fmt(val_metrics, "val_mcc_epoch"),
                _fmt(val_metrics, "val_mae_epoch"),
                _fmt(val_metrics, "val_r2_epoch"),
                "", "", "", "", "", "",
                cfg_copy["LR"], cfg_copy["BATCH_SIZE"],
            ]
        
        if cfg_copy.get("sample_size", None):
            row += [cfg_copy["sample_size"]]

        if os.getenv("SLURM_PROCID", "0") == "0":
            _write_row(cfg_copy["result_path"], header, row)
        logging.info("RESULT_ROW  " + ", ".join(map(str, row)))


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO)
    if len(sys.argv) != 2:
        logging.error("Usage: python -m finetune_runner CONFIG.json")
        sys.exit(1)
    main(sys.argv[1])
