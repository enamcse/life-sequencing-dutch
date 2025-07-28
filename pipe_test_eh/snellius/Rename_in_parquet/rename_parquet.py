#!/usr/bin/env python3
import os, json, argparse
import pandas as pd

def apply_ops(df, ops):
    # 1) rename
    if "rename" in ops:
        df = df.rename(columns=ops["rename"])
    # 2) dtype
    for c, dt in ops.get("dtype", {}).items():
        if c in df.columns:
            df[c] = df[c].astype(dt)
    # 3) drop
    drop_list = [c for c in ops.get("drop", []) if c in df.columns]
    if drop_list:
        df = df.drop(columns=drop_list)
    return df, drop_list

def process_directory(root, cfg):
    global_ops = cfg.get("operations", {})
    file_ops   = cfg.get("file_operations", {})

    for dirpath, _, files in os.walk(root):
        for fname in files:
            if not fname.endswith(".parquet"):
                continue
            if fname.endswith("_meta.parquet"):
                continue  # skip meta here

            data_path = os.path.join(dirpath, fname)
            base      = fname[:-8]  # strip ".parquet"
            meta_path = os.path.join(dirpath, f"{base}_meta.parquet")

            print(f"\n▶ Processing {data_path}")
            df = pd.read_parquet(data_path)

            # determine ops: start with global, then overlay file‑specific
            ops = {
                "rename": dict(global_ops.get("rename", {})),
                "dtype": dict(global_ops.get("dtype", {})),
                "drop": list(global_ops.get("drop", []))
            }
            if fname in file_ops:
                # override/extend
                fp = file_ops[fname]
                ops["rename"].update(fp.get("rename", {}))
                ops["dtype"].update(fp.get("dtype", {}))
                ops["drop"].extend(fp.get("drop", []))

            # apply to data
            df_new, dropped = apply_ops(df, ops)
            df_new.to_parquet(data_path, index=False)
            print(f"   • data updated, dropped columns: {dropped}")

            # now meta
            if os.path.isfile(meta_path):
                meta = pd.read_parquet(meta_path)
                # rename Name entries
                rename_map = ops["rename"]
                mask = meta["Name"].isin(rename_map.keys())
                if mask.any():
                    meta.loc[mask, "Name"] = meta.loc[mask, "Name"].map(rename_map)
                # drop rows for dropped columns
                if dropped:
                    meta = meta[~meta["Name"].isin(dropped)]
                meta.to_parquet(meta_path, index=False)
                print(f"   • meta updated")
            else:
                print(f"   ⚠️  meta file not found at {meta_path}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", required=True, help="Path to JSON config")
    args = p.parse_args()

    cfg = json.load(open(args.cfg))
    for d in cfg.get("target_dirs", []):
        if os.path.isdir(d):
            process_directory(d, cfg)
        else:
            print(f"⚠️  target_dirs entry not found or not a dir: {d}")

if __name__ == "__main__":
    main()
