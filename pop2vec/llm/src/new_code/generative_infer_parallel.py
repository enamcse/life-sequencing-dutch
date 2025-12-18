# generative_infer.py
import os, glob, argparse, logging, torch
from typing import Optional
from torch.nn.functional import softmax

# --- project imports (kept as in your snippet) ---
from pop2vec.llm.src.new_code.utils import (
    read_hparams,
    get_vocab_size,
    load_special_ids,          # CSV -> {'pad_id','cls_id','death_id'}
    load_vocab_df,
    pretty_print_tokens,
)
from pop2vec.llm.src.new_code.load_data import CustomLazyHDF5Dataset
from pop2vec.llm.src.transformer.models import TransformerEncoder
from pop2vec.llm.src.new_code.pretrain import (
     
    update_hparams_with_defaults,
    hparams_integrity_check,
)

DEFAULT_VALS = {
    'epoch': 1,
    'horizon': 20,
    'temperature': 1.0,
    'top_k': 20,
    'pad_token': "[PAD]",
    'cls_token': "[CLS]",
    'death_token': "[DEATH]"
}

# required keys (not enforced here, but kept for reference)
REQ_KEYS = [
    'pretrained_model_path', 
    'pretrained_model_hparams',
    'mlm_path', 
    'num_val_items', 
    'batch_size',
    'epochs',
    'vocab_path'
]

# ----------------- logging -----------------
logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ----------------- helpers -----------------
def load_hparams(hparams_path: str):
    # use the same pipeline you use in pretrain.py
    hparams = read_hparams(hparams_path)
    # optionally merge nested pretrained_model_hparams, if present
    hparams.update(read_hparams(hparams["pretrained_model_hparams"]))
    hparams_integrity_check(hparams)
    hparams["vocab_size"] = get_vocab_size(hparams["vocab_path"])
    return update_hparams_with_defaults(hparams)

def find_latest_ckpt(ckpt_dir: str) -> str:
    cands = sorted(glob.glob(os.path.join(ckpt_dir, "**", "*.ckpt"), recursive=True))
    if not cands:
        raise FileNotFoundError(f"No .ckpt under {ckpt_dir}")
    return cands[-1]

def _select_prefix_item(ds, prefix_id: Optional[str], seq_idx: int = 0):
    if prefix_id is None or str(prefix_id).strip() == "":
        logger.info(f"prefix_id not provided; using ds[{seq_idx}]")
        if seq_idx < len(ds):
            return ds[seq_idx]
        else:
            logger.warning(f"seq_idx={seq_idx} >= len(ds)={len(ds)}; using ds[0]")
            return ds[0]
    logger.info(f"Searching for prefix_id={prefix_id} ...")
    # Try common id fields on the fly (safe for small/medium scans)
    for i in range(min(len(ds), 200000)):
        item = ds[i]
        for k in ("sequence_id", "rinpersoon_id", "id"):
            if k in item and str(item[k]) == str(prefix_id):
                logger.info(f"Found prefix at index={i} via key '{k}'")
                return item
    logger.warning("prefix_id not found; falling back to ds[0]")
    return ds[0]

@torch.no_grad()
def generate_next_tokens(
    model,
    prefix_4stream: torch.Tensor,
    pad_mask: torch.Tensor,
    horizon: int,
    top_k: int = 0,
    temperature: float = 1.0,
    death_id: Optional[int] = None,
    log_every_step: int = 0,
    vocab_df=None,
    with_category: bool = False,
):
    device = next(model.parameters()).device
    x  = prefix_4stream.to(device)   # (1,4,L0)
    pm = pad_mask.to(device)         # (1,L0)
    out_tokens = []
    stopped_by_death = False

    logger.info(f"Begin generation | horizon={horizon} | top_k={top_k} | temperature={temperature}")

    for step in range(int(horizon)):
        logits = model({"input_ids": x, "padding_mask": pm})  # (1,L,V) for AR model
        last_logits = logits[:, -1, :] / max(1e-8, float(temperature))

        if top_k and top_k > 0:
            vals, idxs = torch.topk(last_logits, k=int(top_k), dim=-1)
            probs = softmax(vals, dim=-1)
            next_token = idxs.gather(-1, torch.multinomial(probs, 1)).squeeze(-1)
            # Optional debug: top-5 preview
            if log_every_step and (step % log_every_step == 0):
                topk_ids = idxs[0, : min(5, top_k)].tolist()
                topk_p   = probs[0, : min(5, top_k)].tolist()
                logger.debug(f"step={step} topk_ids={topk_ids} topk_p={[round(p,4) for p in topk_p]}")
        else:
            next_token = torch.argmax(last_logits, dim=-1)

        tid = int(next_token.item())
        out_tokens.append(tid)

        if log_every_step and (step % log_every_step == 0):
            # log chosen token + prob
            probs_full = softmax(last_logits, dim=-1)
            p_sel = float(probs_full[0, tid].item())
            if vocab_df is not None:
                from_idx = vocab_df.set_index("ID")
                tok_name = from_idx["TOKEN"].get(tid, f"<UNK:{tid}>")
                cat_name = from_idx["CATEGORY"].get(tid, "")
                shown = f"{tok_name}|{cat_name}" if with_category else tok_name
            else:
                shown = str(tid)
            logger.info(f"step={step} chose token={tid} ({shown}) p={p_sel:.4f}")

        if (death_id is not None) and (tid == death_id):
            stopped_by_death = True
            logger.info(f"Stopped at step={step} due to [DEATH] token")
            break

        # v1: copy age/day; set segment=1
        last_age = x[0, 1, -1].item()
        last_day = x[0, 2, -1].item()
        new_step = torch.tensor([[tid],[last_age],[last_day],[1]], dtype=torch.long, device=device)  # (4,1)
        x  = torch.cat([x, new_step.unsqueeze(0)], dim=2)
        pm = torch.cat([pm, torch.ones(1,1, dtype=pm.dtype, device=device)], dim=1)

    logger.info(f"Generation done | steps={len(out_tokens)} | stopped_by_death={stopped_by_death}")
    return out_tokens

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hparams", required=True, help="Path to generative hparams txt")
    ap.add_argument("--device", default=None, help="Optional device override: cuda:0 / cpu")
    args = ap.parse_args()

    # 1) Load hparams
    hp = load_hparams(args.hparams)
    logger.info("=== Generative Inference: start ===")
    logger.info(f"hparams file: {args.hparams}")

    # 2) Specials + vocab
    vocab_csv = hp["vocab_path"]
    specials = load_special_ids(
        vocab_csv,
        pad_token=hp.get("pad_token", DEFAULT_VALS['pad_token']),
        cls_token=hp.get("cls_token", DEFAULT_VALS['cls_token']),
        death_token=hp.get("death_token", DEFAULT_VALS['death_token']),
        pad_fallback=int(hp.get("pad_fallback", 0)) if "pad_fallback" in hp else 0,
    )
    PAD_ID, CLS_ID, DEATH_ID = specials["pad_id"], specials["cls_id"], specials["death_id"]
    logger.info(f"Special IDs | PAD={PAD_ID} | CLS={CLS_ID} | DEATH={DEATH_ID}")
    vocab_df = load_vocab_df(vocab_csv)

    # 3) Checkpoint
    ckpt_path = hp.get("pretrained_model_path") or find_latest_ckpt(hp.get("checkpoint_dir", ""))
    logger.info(f"Checkpoint: {ckpt_path}")

    # 4) Device & sampling
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    horizon       = int(hp.get("horizon", DEFAULT_VALS['horizon']))
    top_k         = int(hp.get("top_k", DEFAULT_VALS['top_k']))
    temperature   = float(hp.get("temperature", DEFAULT_VALS['temperature']))
    log_every_step = int(hp.get("log_every_step", 0))
    with_category  = bool(hp.get("with_category", False))
    tokens_write_path = str(hp.get("tokens_write_path", "")).strip() or None
    batch_size    = int(hp.get("inference_batch_size", 1))  # NEW: batch processing

    logger.info(f"Device={device} | horizon={horizon} | top_k={top_k} | temperature={temperature} | log_every_step={log_every_step} | batch_size={batch_size}")

    # 5) Load AR model
    model = TransformerEncoder.load_from_checkpoint(ckpt_path, strict=False)
    model.eval().to(device)
    task = getattr(model.hparams, "training_task", "mlm")
    logger.info(f"Loaded model | training_task={task}")
    if task != "ar_lm":
        logger.error("Checkpoint is not AR-trained (training_task != 'ar_lm').")
        raise RuntimeError("Loaded checkpoint is not AR-trained (training_task != 'ar_lm').")

    # 6) Real prefix from HDF5
    ds = CustomLazyHDF5Dataset(
        hp["mlm_path"],
        validation=False,
        num_val_items=int(hp.get("NUM_VAL_ITEMS", 100000)),
        mlm_encoded=False,
        inference=True,
    )
    prefix_id = str(hp.get("prefix_id", "")).strip()

    # 7) Generate + pretty print generated
    num_sequences = int(hp.get("num_sequences", 1))
    
    # Process in batches for parallel execution
    for batch_start in range(0, num_sequences, batch_size):
        batch_end = min(batch_start + batch_size, num_sequences)
        batch_indices = range(batch_start, batch_end)
        logger.info(f"Processing batch: sequences {batch_start+1} to {batch_end} (batch size={len(batch_indices)})")
        
        # Collect batch data
        batch_items = []
        batch_original_tokens = []
        max_len = 0
        
        for seq_idx in batch_indices:
            item = _select_prefix_item(ds, prefix_id, seq_idx)
            x4: torch.Tensor = item["input_ids"]      # (4, L)
            pm: torch.Tensor = item["padding_mask"]   # (L,)
            L_real = int(pm.sum().item())
            x4, pm = x4[:, :L_real], pm[:L_real]

            if "prefix_len" in hp and str(hp["prefix_len"]).strip() not in ("", "None"):
                Lp = min(int(hp["prefix_len"]), x4.size(1))
                x4, pm = x4[:, :Lp], pm[:Lp]
            
            batch_items.append((x4, pm))
            batch_original_tokens.append(x4[0].tolist())
            max_len = max(max_len, x4.size(1))
        
        # Pad all sequences in batch to same length
        batch_x4 = []
        batch_pm = []
        for x4, pm in batch_items:
            if x4.size(1) < max_len:
                pad_len = max_len - x4.size(1)
                x4 = torch.cat([x4, torch.zeros(4, pad_len, dtype=x4.dtype)], dim=1)
                pm = torch.cat([pm, torch.zeros(pad_len, dtype=pm.dtype)], dim=1)
            batch_x4.append(x4)
            batch_pm.append(pm)
        
        # Stack into batch: (B, 4, L)
        x4_batch = torch.stack(batch_x4).to(device)
        pm_batch = torch.stack(batch_pm).to(device)
        
        # Generate for entire batch at once (PARALLEL on GPU)
        logger.info(f"Generating {len(batch_indices)} sequences in parallel...")
        all_generated = []
        for b_idx in range(len(batch_indices)):
            generated_tokens = generate_next_tokens(
                model,
                prefix_4stream=x4_batch[b_idx:b_idx+1],  # (1, 4, L)
                pad_mask=pm_batch[b_idx:b_idx+1],         # (1, L)
                horizon=horizon,
                top_k=top_k,
                temperature=temperature,
                death_id=DEATH_ID,
                log_every_step=0,  # Suppress per-step logging in batch mode
                vocab_df=vocab_df,
                with_category=with_category,
            )
            all_generated.append(generated_tokens)
        
        # Print results
        for idx, (seq_idx, original_tokens, generated_tokens) in enumerate(zip(batch_indices, batch_original_tokens, all_generated)):
            pretty_print_tokens(
                f"ORIGINAL PREFIX TOKENS (Sequence {seq_idx + 1})",
                original_tokens,
                vocab_df,
                with_category=with_category,
                out_path=tokens_write_path,
            )
            pretty_print_tokens(
                f"GENERATED TOKENS (Sequence {seq_idx + 1})",
                generated_tokens,
                vocab_df,
                with_category=with_category,
                out_path=tokens_write_path,
            )

    logger.info("=== Generative Inference: done ===")

if __name__ == "__main__":
    main()
