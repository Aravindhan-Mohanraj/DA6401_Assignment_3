"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └┘

W&B logging overview (for Part 2 experiments):
  Per step (train):
    train/loss               — cross-entropy / label-smoothing loss
    train/learning_rate      — current LR (exp 2.1: Noam vs fixed)
    train/prediction_confidence — softmax prob of correct token (exp 2.5)
    grad_norm/Q_weight       — mean ‖∇W_q‖ across all encoder layers (exp 2.2)
    grad_norm/K_weight       — mean ‖∇W_k‖ across all encoder layers (exp 2.2)
    train/grad_norm_total    — global gradient norm
    train/step               — global step counter

  Per epoch:
    train/epoch_loss         — average train loss for the epoch
    val/epoch_loss           — average val loss for the epoch
    val/bleu                 — corpus-level BLEU on validation set
    epoch                    — epoch index

  End of run:
    test/bleu                — corpus-level BLEU on held-out test set

  Rich logs:
    translations             — wandb.Table (src, reference, hypothesis)
    attention/head_N         — wandb.Image heatmaps per encoder head (exp 2.3)
    lr_schedule              — wandb.Image of the Noam LR curve
"""

import json
import os
import math
import time
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

try:
    from sacrebleu.metrics import BLEU as SacreBLEU
    _SACREBLEU_AVAILABLE = True
except ImportError:
    _SACREBLEU_AVAILABLE = False

from model import Transformer, make_src_mask, make_tgt_mask
from dataset import (
    Multi30kDataset, build_datasets, collate_fn,
    PAD_IDX, SOS_IDX, EOS_IDX,
)
from lr_scheduler import NoamScheduler, get_lr_history


#  Module-level global step counter (reset at each experiment) 
_GLOBAL_STEP: int = 0


#  LABEL SMOOTHING LOSS

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need"

    Smoothed target distribution:
        y_smooth[t] = (1 - eps)   if t == target
                    = eps / (vocab_size - 1)   otherwise
        y_smooth[pad_idx row] = 0  (pad tokens ignored)

    Args:
        vocab_size (int)  : Number of output classes.
        pad_idx    (int)  : Index of <pad> token — receives 0 probability.
        smoothing  (float): Smoothing factor ε (default 0.1).
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : [batch * tgt_len, vocab_size]  (raw model output)
            target : [batch * tgt_len]              (gold token indices)

        Returns:
            Scalar loss (mean over non-pad tokens).
        """
        # Build smoothed distribution: eps/(V-1) everywhere
        smooth_dist = torch.full_like(
            logits, self.smoothing / max(self.vocab_size - 1, 1)
        )
        # Put (1-eps) mass on the correct token
        smooth_dist.scatter_(1, target.unsqueeze(1), self.confidence)

        # Zero out rows where target is <pad>
        pad_mask = target == self.pad_idx
        smooth_dist[pad_mask] = 0.0

        # KL-style loss: -sum(y_smooth * log_softmax(logits))
        log_probs = F.log_softmax(logits, dim=-1)
        loss_per_token = -(smooth_dist * log_probs).sum(dim=-1)

        # Average only over non-pad positions
        n_tokens = (~pad_mask).sum().clamp(min=1).float()
        return loss_per_token.sum() / n_tokens


#  TRAINING LOOP

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    """
    Run one epoch of training or evaluation.

    W&B metrics logged per step (train only):
        train/loss, train/learning_rate, train/prediction_confidence,
        grad_norm/Q_weight, grad_norm/K_weight, train/grad_norm_total

    Args:
        data_iter  : DataLoader yielding (src, tgt) token-index tensors.
        model      : Transformer instance.
        loss_fn    : LabelSmoothingLoss (or any nn.Module loss).
        optimizer  : Optimizer (None during eval).
        scheduler  : NoamScheduler instance (None during eval).
        epoch_num  : Current epoch index (for logging).
        is_train   : If True, perform backward pass and scheduler step.
        device     : 'cpu' or 'cuda'.

    Returns:
        avg_loss : Average loss over the epoch (float).
    """
    global _GLOBAL_STEP

    model.train() if is_train else model.eval()

    total_loss   = 0.0
    n_batches    = 0
    use_wandb    = _WANDB_AVAILABLE and wandb.run is not None
    mode_str     = "Train" if is_train else "Val"

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    from tqdm import tqdm
    pbar = tqdm(
        data_iter,
        desc=f"Epoch {epoch_num:3d} [{mode_str}]",
        leave=False,
        dynamic_ncols=True,
    )

    with ctx:
        for batch_idx, (src, tgt) in enumerate(pbar):
            src = src.to(device)   # [B, src_len]
            tgt = tgt.to(device)   # [B, tgt_len]

            # Teacher forcing: input = tgt[:-1], label = tgt[1:]
            tgt_input  = tgt[:, :-1]   # [B, T-1]
            tgt_output = tgt[:, 1:]    # [B, T-1]

            src_mask = make_src_mask(src)          # [B,1,1,src_len]
            tgt_mask = make_tgt_mask(tgt_input)    # [B,1,T-1,T-1]

            logits = model(src, tgt_input, src_mask, tgt_mask)
            # logits: [B, T-1, vocab_size]

            B, T, V = logits.shape
            loss = loss_fn(logits.view(-1, V), tgt_output.reshape(-1))

            if is_train:
                optimizer.zero_grad()
                loss.backward()

                #  Gradient norms (exp 2.2) 
                q_norms, k_norms = [], []
                for name, param in model.named_parameters():
                    if param.grad is None:
                        continue
                    if "W_q.weight" in name:
                        q_norms.append(param.grad.norm().item())
                    if "W_k.weight" in name:
                        k_norms.append(param.grad.norm().item())

                q_grad_norm = float(np.mean(q_norms)) if q_norms else 0.0
                k_grad_norm = float(np.mean(k_norms)) if k_norms else 0.0

                # Global gradient norm + clipping
                total_grad_norm = nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                ).item()

                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                #  Prediction confidence (exp 2.5) 
                with torch.no_grad():
                    probs = F.softmax(logits.view(-1, V), dim=-1)
                    gold  = tgt_output.reshape(-1)
                    correct_probs = probs.gather(1, gold.unsqueeze(1)).squeeze(1)
                    non_pad = gold != PAD_IDX
                    confidence = correct_probs[non_pad].mean().item() if non_pad.any() else 0.0

                #  W&B per-step logging 
                if use_wandb:
                    current_lr = (
                        scheduler.get_last_lr()[0]
                        if scheduler is not None
                        else optimizer.param_groups[0]["lr"]
                    )
                    wandb.log(
                        {
                            "train/loss":                   loss.item(),
                            "train/learning_rate":          current_lr,
                            "train/prediction_confidence":  confidence,
                            "grad_norm/Q_weight":           q_grad_norm,
                            "grad_norm/K_weight":           k_grad_norm,
                            "train/grad_norm_total":        total_grad_norm,
                            "train/step":                   _GLOBAL_STEP,
                            "epoch":                        epoch_num,
                        },
                        step=_GLOBAL_STEP,
                    )

                _GLOBAL_STEP += 1

            total_loss += loss.item()
            n_batches  += 1

            # Update progress bar with running average loss
            pbar.set_postfix({"loss": f"{total_loss / n_batches:.4f}"})

    pbar.close()
    return total_loss / max(n_batches, 1)


#  GREEDY DECODING

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int = EOS_IDX,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.

    Args:
        model        : Trained Transformer.
        src          : [1, src_len] token indices.
        src_mask     : [1, 1, 1, src_len].
        max_len      : Maximum number of tokens to generate.
        start_symbol : Vocabulary index of <sos>.
        end_symbol   : Vocabulary index of <eos>.
        device       : 'cpu' or 'cuda'.

    Returns:
        ys : [1, out_len] — includes start_symbol; stops at end_symbol
             or when max_len is reached.
    """
    model.eval()
    with torch.no_grad():
        memory = model.encode(src, src_mask)         # [1, src_len, d_model]
        ys = torch.tensor([[start_symbol]], device=device)  # [1, 1]

        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys)             # [1, 1, cur_len, cur_len]
            logits   = model.decode(memory, src_mask, ys, tgt_mask)
            # [1, cur_len, vocab_size] — take last position
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [1, 1]
            ys = torch.cat([ys, next_tok], dim=1)    # [1, cur_len+1]

            if next_tok.item() == end_symbol:
                break

    return ys                                        # [1, out_len]


#  BLEU EVALUATION

def _tokens_to_str(ids, vocab, skip_ids) -> str:
    """Convert a list of token indices to a detokenised string."""
    return " ".join(
        vocab.itos[i] for i in ids if i not in skip_ids
    )


def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score (0–100).

    Args:
        model           : Trained Transformer (put in eval mode internally).
        test_dataloader : DataLoader yielding (src, tgt) token-index tensors.
        tgt_vocab       : Vocab object (supports .itos[idx] / .lookup_token(idx)).
        device          : 'cpu' or 'cuda'.
        max_len         : Max decode length per sentence.

    Returns:
        bleu_score : Corpus-level BLEU (float, 0–100).
    """
    model.eval()

    sos_idx = tgt_vocab.stoi.get("<sos>", SOS_IDX)
    eos_idx = tgt_vocab.stoi.get("<eos>", EOS_IDX)
    pad_idx = tgt_vocab.stoi.get("<pad>", PAD_IDX)
    skip    = {sos_idx, eos_idx, pad_idx}

    hypotheses: list[str] = []
    references: list[str] = []

    with torch.no_grad():
        for src_batch, tgt_batch in test_dataloader:
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            for i in range(src_batch.size(0)):
                src_i    = src_batch[i : i + 1]              # [1, src_len]
                src_mask = make_src_mask(src_i)

                # Greedy decode
                pred = greedy_decode(
                    model, src_i, src_mask, max_len,
                    sos_idx, eos_idx, device,
                )
                pred_ids = pred.squeeze(0).tolist()

                # Strip <sos>/<eos>/<pad>
                hyp = _tokens_to_str(pred_ids, tgt_vocab, skip)

                # Reference
                ref_ids = tgt_batch[i].tolist()
                ref = _tokens_to_str(ref_ids, tgt_vocab, skip)

                hypotheses.append(hyp)
                references.append(ref)

    # Compute corpus BLEU
    if _SACREBLEU_AVAILABLE:
        metric = SacreBLEU(tokenize="none", force=True)
        score  = metric.corpus_score(hypotheses, [references]).score
    else:
        # Fallback: sentence-average BLEU via NLTK if sacrebleu missing
        try:
            from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
            refs_tok  = [[ref.split()] for ref in references]
            hyps_tok  = [hyp.split() for hyp in hypotheses]
            smooth    = SmoothingFunction().method1
            score     = corpus_bleu(refs_tok, hyps_tok, smoothing_function=smooth) * 100
        except ImportError:
            # Last resort: just count exact matches (very rough)
            score = sum(h == r for h, r in zip(hypotheses, references)) / max(len(hypotheses), 1) * 100

    return float(score)


#  CHECKPOINT UTILITIES

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model + optimiser + scheduler state to disk.

    Saved dict keys:
        'epoch', 'model_state_dict', 'optimizer_state_dict',
        'scheduler_state_dict', 'model_config'
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save(
        {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (
                scheduler.state_dict() if scheduler is not None else {}
            ),
            "model_config":         model._model_config,
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) state from disk.

    Args:
        path      : Path to checkpoint file saved by save_checkpoint.
        model     : Transformer with matching architecture (state loaded in-place).
        optimizer : Optimizer to restore (pass None to skip).
        scheduler : Scheduler to restore (pass None to skip).

    Returns:
        epoch : The epoch at which the checkpoint was saved (int).
    """
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    return int(ckpt["epoch"])


#  PART 2 HELPERS — W&B visualisations

def _log_lr_schedule(d_model: int, warmup_steps: int, total_steps: int = 20_000) -> None:
    """Log a Noam LR schedule plot to W&B."""
    if not (_WANDB_AVAILABLE and wandb.run is not None):
        return
    from lr_scheduler import get_lr_history
    lrs = get_lr_history(d_model, warmup_steps, total_steps)
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(lrs)
    ax.axvline(warmup_steps, color="red", linestyle="--", label=f"warmup={warmup_steps}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title(f"Noam LR Schedule (d_model={d_model})")
    ax.legend()
    fig.tight_layout()
    wandb.log({"lr_schedule": wandb.Image(fig)}, step=0)
    plt.close(fig)


def _log_attention_heatmaps(
    model: Transformer,
    sample_src: torch.Tensor,
    src_vocab,
    device: str,
) -> None:
    """
    Extract last-encoder-layer attention weights for sample_src and log
    one heatmap per head to W&B.  (Experiment 2.3)

    Args:
        model      : Trained Transformer.
        sample_src : [1, src_len] tensor of source token indices.
        src_vocab  : Vocab object for label lookup.
        device     : device string.
    """
    if not (_WANDB_AVAILABLE and wandb.run is not None):
        return

    model.eval()
    sample_src = sample_src.to(device)

    with torch.no_grad():
        src_mask = make_src_mask(sample_src)
        model.encode(sample_src, src_mask)

    attn = model.get_last_encoder_attn()   # [1, H, src_len, src_len]
    if attn is None:
        return

    attn = attn[0].cpu().numpy()           # [H, src_len, src_len]
    num_heads = attn.shape[0]

    # Build token labels (strip pad)
    ids     = sample_src[0].cpu().tolist()
    labels  = [src_vocab.itos[i] for i in ids if i != PAD_IDX]
    src_len = len(labels)

    for h in range(num_heads):
        weights = attn[h, :src_len, :src_len]
        fig, ax = plt.subplots(figsize=(max(6, src_len * 0.5), max(5, src_len * 0.45)))
        im = ax.imshow(weights, cmap="viridis", aspect="auto", vmin=0, vmax=weights.max())
        ax.set_xticks(range(src_len))
        ax.set_yticks(range(src_len))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_title(f"Encoder Last Layer — Head {h + 1}")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        wandb.log({f"attention/head_{h + 1}": wandb.Image(fig)}, commit=(h == num_heads - 1))
        plt.close(fig)


def _log_translation_table(
    model: Transformer,
    src_batch: torch.Tensor,
    tgt_batch: torch.Tensor,
    src_vocab,
    tgt_vocab,
    device: str,
    n: int = 5,
    epoch: int = 0,
    step: int = 0,
) -> None:
    """Log a W&B Table of (source, reference, hypothesis) for n examples."""
    if not (_WANDB_AVAILABLE and wandb.run is not None):
        return

    sos = tgt_vocab.stoi.get("<sos>", SOS_IDX)
    eos = tgt_vocab.stoi.get("<eos>", EOS_IDX)
    pad = tgt_vocab.stoi.get("<pad>", PAD_IDX)
    skip_tgt = {sos, eos, pad}
    skip_src = {src_vocab.stoi.get("<sos>", SOS_IDX),
                src_vocab.stoi.get("<eos>", EOS_IDX),
                src_vocab.stoi.get("<pad>", PAD_IDX)}

    table = wandb.Table(columns=["epoch", "source (de)", "reference (en)", "hypothesis (en)"])

    model.eval()
    src_batch = src_batch.to(device)
    tgt_batch = tgt_batch.to(device)

    with torch.no_grad():
        for i in range(min(n, src_batch.size(0))):
            src_i    = src_batch[i : i + 1]
            src_mask = make_src_mask(src_i)

            pred = greedy_decode(model, src_i, src_mask, 80, sos, eos, device)
            pred_ids = pred.squeeze(0).tolist()

            src_str  = _tokens_to_str(src_batch[i].tolist(), src_vocab, skip_src)
            ref_str  = _tokens_to_str(tgt_batch[i].tolist(), tgt_vocab, skip_tgt)
            hyp_str  = _tokens_to_str(pred_ids, tgt_vocab, skip_tgt)

            table.add_data(epoch, src_str, ref_str, hyp_str)

    wandb.log({"translations": table}, step=step)


#  TRAINING SUMMARY HELPER

def _save_training_summary(
    config: dict,
    best_val_bleu: float,
    best_ckpt_path: str,
    run_name: str,
    best_epoch: Optional[int] = None,
    test_bleu: Optional[float] = None,
) -> None:
    """
    Save a human-readable JSON summary of the current best training state.
    Written to training_summary.json in the project directory.
    Updated every time a new best val BLEU is found, and once more at the
    end with the final test BLEU.
    """
    summary = {
        "run_name":        run_name,
        "best_val_bleu":   round(best_val_bleu, 4),
        "best_checkpoint": best_ckpt_path,
        "config":          config,
    }
    if best_epoch is not None:
        summary["best_epoch"] = best_epoch
    if test_bleu is not None:
        summary["test_bleu"] = round(test_bleu, 4)

    with open("training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


#  EXPERIMENT ENTRY POINT

def run_training_experiment(
    #  Architecture 
    d_model: int   = 256,
    N: int         = 3,
    num_heads: int = 8,
    d_ff: int      = 512,
    dropout: float = 0.1,
    #  Training 
    batch_size: int  = 128,
    num_epochs: int  = 15,
    warmup_steps: int = 4000,
    #  Experiment flags (Part 2) 
    label_smoothing: float = 0.1,     # exp 2.5: 0.1 or 0.0
    scheduler_type: str   = "noam",   # exp 2.1: 'noam' or 'fixed'
    fixed_lr: float       = 1e-4,     # exp 2.1: LR when scheduler_type='fixed'
    use_scaling: bool     = True,     # exp 2.2: True or False
    pe_type: str          = "sinusoidal",  # exp 2.4: 'sinusoidal' or 'learned'
    #  I/O 
    checkpoint_path: str  = "checkpoints/checkpoint.pt",
    best_ckpt_path: str   = "checkpoints/best_checkpoint.pt",
    device: str           = None,
    project_name: str     = "da6401-a3",
    run_name: str         = None,
) -> None:
    """
    Set up and run the full training experiment.

    All hyperparameters and experiment flags are exposed as arguments so
    each Part 2 experiment can be launched by changing a single flag.
    """
    global _GLOBAL_STEP
    _GLOBAL_STEP = 0

    #  Device 
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    #  W&B init 
    config = {
        "d_model":          d_model,
        "N":                N,
        "num_heads":        num_heads,
        "d_ff":             d_ff,
        "dropout":          dropout,
        "batch_size":       batch_size,
        "num_epochs":       num_epochs,
        "warmup_steps":     warmup_steps,
        "label_smoothing":  label_smoothing,
        "scheduler_type":   scheduler_type,
        "fixed_lr":         fixed_lr if scheduler_type == "fixed" else None,
        "use_scaling":      use_scaling,
        "pe_type":          pe_type,
        "optimizer":        "Adam",
        "adam_beta1":       0.9,
        "adam_beta2":       0.98,
        "adam_eps":         1e-9,
        "grad_clip":        1.0,
        "dataset":          "Multi30k (de→en)",
    }

    if _WANDB_AVAILABLE:
        wandb.init(
            project=project_name,
            name=run_name,
            config=config,
        )
        # Log LR schedule plot upfront for Noam runs
        if scheduler_type == "noam":
            _log_lr_schedule(d_model, warmup_steps)

    #  Dataset & vocab 
    print("[INFO] Loading Multi30k dataset …")
    train_ds, val_ds, test_ds = build_datasets(min_freq=1)

    src_vocab = train_ds.src_vocab
    tgt_vocab = train_ds.tgt_vocab

    if _WANDB_AVAILABLE and wandb.run is not None:
        wandb.config.update({
            "src_vocab_size": len(src_vocab),
            "tgt_vocab_size": len(tgt_vocab),
        })

    print(f"[INFO] src vocab size: {len(src_vocab)}  tgt vocab size: {len(tgt_vocab)}")
    print(f"[INFO] train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    _collate = partial(collate_fn, pad_idx=PAD_IDX)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=_collate, num_workers=0, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=_collate, num_workers=0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=_collate, num_workers=0,
    )

    #  Model 
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=d_model,
        N=N,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=dropout,
        pe_type=pe_type,
        use_scaling=use_scaling,
        load_pretrained=False,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Trainable parameters: {n_params:,}")
    if _WANDB_AVAILABLE and wandb.run is not None:
        wandb.config.update({"n_params": n_params})

    #  Optimizer 
    init_lr  = 1.0 if scheduler_type == "noam" else fixed_lr
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=init_lr,
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    #  Scheduler 
    if scheduler_type == "noam":
        scheduler = NoamScheduler(optimizer, d_model=d_model, warmup_steps=warmup_steps)
    else:
        scheduler = None   # constant LR

    #  Loss 
    loss_fn = LabelSmoothingLoss(
        vocab_size=len(tgt_vocab),
        pad_idx=PAD_IDX,
        smoothing=label_smoothing,
    )

    #  Training loop 
    best_val_bleu = -1.0
    best_epoch    = 0
    # Grab a fixed batch for the translation table each epoch
    sample_batch = next(iter(val_loader))

    for epoch in range(num_epochs):
        t0 = time.time()

        # Train
        train_loss = run_epoch(
            train_loader, model, loss_fn,
            optimizer, scheduler,
            epoch_num=epoch, is_train=True, device=device,
        )

        # Validate
        val_loss = run_epoch(
            val_loader, model, loss_fn,
            None, None,
            epoch_num=epoch, is_train=False, device=device,
        )

        # BLEU on validation set
        val_bleu = evaluate_bleu(model, val_loader, tgt_vocab, device=device)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_bleu={val_bleu:.2f} | "
            f"time={elapsed:.1f}s"
        )

        #  Per-epoch W&B logging 
        if _WANDB_AVAILABLE and wandb.run is not None:
            wandb.log(
                {
                    "train/epoch_loss": train_loss,
                    "val/epoch_loss":   val_loss,
                    "val/bleu":         val_bleu,
                    "epoch":            epoch,
                    "epoch_time_s":     elapsed,
                },
                step=_GLOBAL_STEP,
            )
            # Translation examples table
            _log_translation_table(
                model,
                sample_batch[0], sample_batch[1],
                src_vocab, tgt_vocab,
                device=device, n=5, epoch=epoch, step=_GLOBAL_STEP,
            )

        # Save checkpoint every epoch
        save_checkpoint(model, optimizer, scheduler, epoch, path=checkpoint_path)

        # Save best model
        if val_bleu > best_val_bleu:
            best_val_bleu = val_bleu
            best_epoch    = epoch
            save_checkpoint(model, optimizer, scheduler, epoch, path=best_ckpt_path)
            print(f"  → New best val BLEU: {best_val_bleu:.2f}  (saved to {best_ckpt_path})")

            # Keep a human-readable record of the current best
            _save_training_summary(
                config=config,
                best_epoch=epoch,
                best_val_bleu=best_val_bleu,
                best_ckpt_path=best_ckpt_path,
                run_name=run_name,
            )

    #  Attention heatmaps (exp 2.3) — on best model 
    print("[INFO] Loading best checkpoint for attention visualisation …")
    load_checkpoint(best_ckpt_path, model)
    # Use first sentence of val set as the sample
    sample_src = sample_batch[0][0:1]
    _log_attention_heatmaps(model, sample_src, src_vocab, device)

    #  Final BLEU on test set 
    print("[INFO] Computing test BLEU …")
    test_bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device)
    print(f"[RESULT] Test BLEU: {test_bleu:.2f}")

    if _WANDB_AVAILABLE and wandb.run is not None:
        wandb.log({"test/bleu": test_bleu})
        wandb.summary["test_bleu"]     = test_bleu
        wandb.summary["best_val_bleu"] = best_val_bleu

        # Log best checkpoint as a W&B artifact
        artifact = wandb.Artifact("best_model", type="model")
        artifact.add_file(best_ckpt_path)
        wandb.log_artifact(artifact)

        wandb.finish()

    # Final summary with test BLEU
    _save_training_summary(
        config=config,
        best_epoch=None,
        best_val_bleu=best_val_bleu,
        best_ckpt_path=best_ckpt_path,
        run_name=run_name,
        test_bleu=test_bleu,
    )

    print("[INFO] Training complete.")

    return {
        "run_name":      run_name,
        "best_val_bleu": round(best_val_bleu, 4),
        "best_epoch":    best_epoch,
        "test_bleu":     round(test_bleu, 4),
        "config":        config,
    }


#  MAIN

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DA6401 Assignment 3 — Transformer NMT")
    parser.add_argument("--d_model",         type=int,   default=256)
    parser.add_argument("--N",               type=int,   default=3)
    parser.add_argument("--num_heads",       type=int,   default=8)
    parser.add_argument("--d_ff",            type=int,   default=512)
    parser.add_argument("--dropout",         type=float, default=0.1)
    parser.add_argument("--batch_size",      type=int,   default=128)
    parser.add_argument("--num_epochs",      type=int,   default=15)
    parser.add_argument("--warmup_steps",    type=int,   default=4000)
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Exp 2.5: 0.1 or 0.0")
    parser.add_argument("--scheduler_type",  type=str,   default="noam",
                        choices=["noam", "fixed"],
                        help="Exp 2.1: noam or fixed")
    parser.add_argument("--fixed_lr",        type=float, default=1e-4,
                        help="Exp 2.1: LR when scheduler_type=fixed")
    parser.add_argument("--use_scaling",     type=lambda x: x.lower() != "false",
                        default=True,
                        help="Exp 2.2: True or False")
    parser.add_argument("--pe_type",         type=str,   default="sinusoidal",
                        choices=["sinusoidal", "learned"],
                        help="Exp 2.4: sinusoidal or learned")
    parser.add_argument("--checkpoint_path", type=str,   default="checkpoints/checkpoint.pt")
    parser.add_argument("--best_ckpt_path",  type=str,   default="checkpoints/best_checkpoint.pt")
    parser.add_argument("--device",          type=str,   default=None)
    parser.add_argument("--project_name",    type=str,   default="da6401-a3")
    parser.add_argument("--run_name",        type=str,   default=None)

    args = parser.parse_args()
    run_training_experiment(**vars(args))
