"""
hyperparameter_search.py — Sequential HP search for DA6401 Assignment 3
Runs 10 configurations, logs each to W&B as HP_tuning_1 ... HP_tuning_10,
saves all results + best config inside the hp_search/ folder.

Usage:
    python hyperparameter_search.py

Outputs (all inside hp_search/):
    hp_results.json      — results of every config after it finishes
    best_config.json     — winning config ready to plug into train.py
    checkpoints/         — best checkpoint per config
"""

import json
import os
from functools import partial

import torch
from torch.utils.data import DataLoader

from train import run_training_experiment, load_checkpoint, evaluate_bleu
from dataset import build_datasets, collate_fn, PAD_IDX
from model import Transformer

# ── Output folder (inside project dir) ────────────────────────────────
RESULTS_DIR = "hp_search"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "checkpoints"), exist_ok=True)

# ── Search settings ───────────────────────────────────────────────────
SCREEN_EPOCHS = 25          # epochs per config (enough to compare fairly)
BATCH_SIZE    = 256
PROJECT       = "da6401-a3"

# ── 10 Configurations ─────────────────────────────────────────────────
# Fixed across all: Noam scheduler, label_smoothing=0.1, sinusoidal PE
# Varying: d_model, N, d_ff, dropout, warmup_steps
CONFIGS = [
    {
        # Reference point — what we already trained
        "d_model": 256, "N": 4, "num_heads": 8, "d_ff": 512,
        "dropout": 0.1, "warmup_steps": 4000,
        "note": "baseline — current best",
    },
    {
        # FFN at the paper ratio (d_ff = 4 × d_model)
        "d_model": 256, "N": 4, "num_heads": 8, "d_ff": 1024,
        "dropout": 0.1, "warmup_steps": 4000,
        "note": "wider FFN (4x d_model ratio)",
    },
    {
        # More layers, wider FFN
        "d_model": 256, "N": 6, "num_heads": 8, "d_ff": 1024,
        "dropout": 0.1, "warmup_steps": 4000,
        "note": "deeper (6 layers) + wider FFN",
    },
    {
        # Bump model dim to 512 — bigger representations
        "d_model": 512, "N": 4, "num_heads": 8, "d_ff": 1024,
        "dropout": 0.1, "warmup_steps": 4000,
        "note": "larger d_model=512, N=4",
    },
    {
        # Bigger model needs more regularisation on small dataset
        "d_model": 512, "N": 4, "num_heads": 8, "d_ff": 1024,
        "dropout": 0.2, "warmup_steps": 4000,
        "note": "d_model=512 + higher dropout to fight overfitting",
    },
    {
        # Full paper base config — designed for large corpora
        "d_model": 512, "N": 6, "num_heads": 8, "d_ff": 2048,
        "dropout": 0.1, "warmup_steps": 8000,
        "note": "original paper base model — more warmup for stability",
    },
    {
        # Paper base + heavier regularisation for Multi30k
        "d_model": 512, "N": 6, "num_heads": 8, "d_ff": 2048,
        "dropout": 0.2, "warmup_steps": 8000,
        "note": "paper base + dropout=0.2 for small dataset",
    },
    {
        # Shallow but very wide — fewer layers, fat FFN
        "d_model": 512, "N": 3, "num_heads": 8, "d_ff": 2048,
        "dropout": 0.1, "warmup_steps": 4000,
        "note": "shallow (3 layers) but wide FFN",
    },
    {
        # More attention heads → finer-grained attention
        "d_model": 512, "N": 4, "num_heads": 16, "d_ff": 1024,
        "dropout": 0.1, "warmup_steps": 4000,
        "note": "16 attention heads (d_k=32 per head)",
    },
    {
        # Longer warmup + deeper model — conservative schedule
        "d_model": 256, "N": 6, "num_heads": 8, "d_ff": 1024,
        "dropout": 0.15, "warmup_steps": 8000,
        "note": "deeper model + long warmup + slight extra dropout",
    },
]


def run_search():
    results   = []
    best_bleu  = -1.0
    best_cfg   = None
    best_rank  = None

    # Load dataset once and reuse across all configs
    print("[INFO] Loading dataset …")
    train_ds, val_ds, _ = build_datasets()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _collate   = partial(collate_fn, pad_idx=PAD_IDX)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, collate_fn=_collate)

    for idx, cfg in enumerate(CONFIGS, start=1):
        run_name   = f"HP_tuning_{idx}"
        ckpt_best  = os.path.join(RESULTS_DIR, "checkpoints", f"hp_{idx}_best.pt")
        ckpt_last  = os.path.join(RESULTS_DIR, "checkpoints", f"hp_{idx}_last.pt")

        print(f"\n{'='*65}")
        print(f"  HP_tuning_{idx} / 10  —  {cfg['note']}")
        print(f"  d_model={cfg['d_model']}  N={cfg['N']}  heads={cfg['num_heads']}"
              f"  d_ff={cfg['d_ff']}  dropout={cfg['dropout']}"
              f"  warmup={cfg['warmup_steps']}")
        print('='*65)

        try:
            run_training_experiment(
                d_model         = cfg["d_model"],
                N               = cfg["N"],
                num_heads       = cfg["num_heads"],
                d_ff            = cfg["d_ff"],
                dropout         = cfg["dropout"],
                batch_size      = BATCH_SIZE,
                num_epochs      = SCREEN_EPOCHS,
                warmup_steps    = cfg["warmup_steps"],
                label_smoothing = 0.1,
                scheduler_type  = "noam",
                use_scaling     = True,
                pe_type         = "sinusoidal",
                checkpoint_path = ckpt_last,
                best_ckpt_path  = ckpt_best,
                run_name        = run_name,
                project_name    = PROJECT,
            )
        except Exception as e:
            print(f"[ERROR] {run_name} failed: {e}")
            results.append({"run": run_name, "val_bleu": 0.0, "error": str(e), **cfg})
            _save_results(results, best_cfg)
            continue

        # ── Evaluate best checkpoint on validation set ─────────────
        val_bleu = 0.0
        if os.path.exists(ckpt_best):
            model = Transformer(
                src_vocab_size = len(train_ds.src_vocab),
                tgt_vocab_size = len(train_ds.tgt_vocab),
                d_model   = cfg["d_model"],
                N         = cfg["N"],
                num_heads = cfg["num_heads"],
                d_ff      = cfg["d_ff"],
                dropout   = cfg["dropout"],
            ).to(device)
            load_checkpoint(ckpt_best, model)
            val_bleu = evaluate_bleu(model, val_loader, train_ds.tgt_vocab, device=device)
            del model
            torch.cuda.empty_cache()

        result = {
            "run":          run_name,
            "val_bleu":     round(val_bleu, 4),
            "d_model":      cfg["d_model"],
            "N":            cfg["N"],
            "num_heads":    cfg["num_heads"],
            "d_ff":         cfg["d_ff"],
            "dropout":      cfg["dropout"],
            "warmup_steps": cfg["warmup_steps"],
            "note":         cfg["note"],
            "checkpoint":   ckpt_best,
        }
        results.append(result)
        print(f"\n  [{run_name}]  Val BLEU = {val_bleu:.2f}")

        if val_bleu > best_bleu:
            best_bleu = val_bleu
            best_cfg  = result.copy()
            best_rank = idx
            print(f"  ★  New best!")

        _save_results(results, best_cfg)

    # ── Final leaderboard ─────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  HYPERPARAMETER SEARCH — FINAL LEADERBOARD")
    print('='*65)
    ranked = sorted(results, key=lambda x: x["val_bleu"], reverse=True)
    for rank, r in enumerate(ranked, 1):
        star = " ★" if r["run"] == best_cfg.get("run") else ""
        print(f"  {rank:2d}. {r['run']:15s}  Val BLEU={r['val_bleu']:.2f}"
              f"  d_model={r['d_model']}  N={r['N']}"
              f"  d_ff={r['d_ff']}  dropout={r['dropout']}{star}")

    print(f"\n  Best: {best_cfg['run']}  (Val BLEU {best_bleu:.2f})")

    # ── Save best config ──────────────────────────────────────────────
    best_out = {
        "d_model":        best_cfg["d_model"],
        "N":              best_cfg["N"],
        "num_heads":      best_cfg["num_heads"],
        "d_ff":           best_cfg["d_ff"],
        "dropout":        best_cfg["dropout"],
        "warmup_steps":   best_cfg["warmup_steps"],
        "batch_size":     BATCH_SIZE,
        "label_smoothing": 0.1,
        "scheduler_type": "noam",
        "val_bleu_at_screening": round(best_bleu, 4),
        "run_name":       best_cfg["run"],
        "note":           best_cfg["note"],
    }
    best_path = os.path.join(RESULTS_DIR, "best_config.json")
    with open(best_path, "w") as f:
        json.dump(best_out, f, indent=2)

    print(f"\n  Saved → {best_path}")
    print(f"\n  Full training command with best config:")
    print(f"  python train.py \\")
    print(f"    --d_model {best_out['d_model']} \\")
    print(f"    --N {best_out['N']} \\")
    print(f"    --num_heads {best_out['num_heads']} \\")
    print(f"    --d_ff {best_out['d_ff']} \\")
    print(f"    --dropout {best_out['dropout']} \\")
    print(f"    --warmup_steps {best_out['warmup_steps']} \\")
    print(f"    --batch_size {BATCH_SIZE} \\")
    print(f"    --num_epochs 100 \\")
    print(f"    --label_smoothing 0.1 \\")
    print(f"    --scheduler_type noam \\")
    print(f"    --run_name best_model_full")

    return best_out


def _save_results(results, best_cfg):
    path = os.path.join(RESULTS_DIR, "hp_results.json")
    # Store as {HP_tuning_1: {...}, HP_tuning_2: {...}, ...}
    results_dict = {r["run"]: {k: v for k, v in r.items() if k != "run"} for r in results}
    with open(path, "w") as f:
        json.dump({"runs": results_dict, "best_so_far": best_cfg}, f, indent=2)


if __name__ == "__main__":
    run_search()
