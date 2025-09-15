#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
import os
from datasets import (
    load_dataset,
    load_from_disk,
    Dataset,
    DatasetDict,
    concatenate_datasets,
)

from utils import (
    SEED,
    FOLDER_PROCESSED,
)

def transform(example):
    """Clean prefixes and strip whitespace from questions and answers."""
    q = re.sub(r"^\s*[Qq]\s*:\s*", "", example.get("question", "")).strip()
    a = re.sub(r"^\s*[Aa]\s*:\s*", "", example.get("answer", "")).strip()
    return {"question": q, "answer": a}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Path to local dataset (load_from_disk) or Hub dataset ID (load_dataset)")
    parser.add_argument("--from-disk", action="store_true", help="Use load_from_disk if source is a local Arrow dataset folder")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--num-proc", type=int, default=1)
    args = parser.parse_args()

    # 1) Load dataset
    ds = load_from_disk(args.source) if args.from_disk else load_dataset(args.source)
    if isinstance(ds, Dataset):  # if no splits provided
        ds = DatasetDict({"train": ds})

    # 2) Transform question/answer
    ds = ds.map(transform, num_proc=args.num_proc, desc="Transforming question/answer")

    # 3) Merge splits (if multiple) and re-split into train/val/test
    base = concatenate_datasets(list(ds.values()))
    tts = base.train_test_split(test_size=args.test_size, seed=args.seed)
    val_ratio = args.val_size / (1.0 - args.test_size) if (1.0 - args.test_size) > 0 else 0.0
    tv = tts["train"].train_test_split(test_size=val_ratio, seed=args.seed)

    final = DatasetDict({
        "train": tv["train"],
        "validation": tv["test"],
        "test": tts["test"],
    })

    # 4) Save to disk in Datasets format
    name_dataset = args.source.split("/")[-1]
    out_dir = os.path.join(FOLDER_PROCESSED, name_dataset)
    final.save_to_disk(out_dir)

    print("✅ Done.")
    print(f"📁 Saved to: {out_dir}")
    print({k: v.num_rows for k, v in final.items()})

if __name__ == "__main__":
    main()

