"""
save_vocab.py — Save trained vocabulary to JSON files.
Run once after training to generate vocab_src.json and vocab_tgt.json,
which are loaded by Transformer.__init__ during inference.

Usage:
    python save_vocab.py
"""

import json
from dataset import build_datasets

print("[INFO] Building vocab from Multi30k training split ...")
train_ds, _, _ = build_datasets(min_freq=1)

src_data = {"stoi": train_ds.src_vocab.stoi, "itos": train_ds.src_vocab.itos}
tgt_data = {"stoi": train_ds.tgt_vocab.stoi, "itos": train_ds.tgt_vocab.itos}

with open("vocab_src.json", "w") as f:
    json.dump(src_data, f)

with open("vocab_tgt.json", "w") as f:
    json.dump(tgt_data, f)

print(f"[DONE] vocab_src.json  ({len(src_data['itos'])} tokens)")
print(f"[DONE] vocab_tgt.json  ({len(tgt_data['itos'])} tokens)")
