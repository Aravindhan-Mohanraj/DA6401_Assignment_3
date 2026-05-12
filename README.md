# **DA6401 - Assignment 3: Implementing the Transformer for Machine Translation**
## ***Aravindhan Mohanraj [DA25S006]***

### WandB report - https://wandb.ai/da25s006-indian-institute-of-technology-madras/da6401-a3/reports/DA6401-Assignment-3--VmlldzoxNjg0Njk1MA?accessToken=9pcen2tr1y9flt8biqm72ufr0cm3kvy527zouyefvviqy1wfywjnci1xefzptotl

### github link - https://github.com/Aravindhan-Mohanraj/DA6401_Assignment_3.git

---

## Problem Statement

Build a Neural Machine Translation (NMT) system that translates text from **German → English** by implementing the Transformer architecture from scratch using PyTorch, following the paper *"Attention Is All You Need"* (Vaswani et al., 2017). The model must support inference on arbitrary German sentences via a single `infer(sentence)` method without reloading the dataset.

---

## Objective

- Implement the full Transformer encoder-decoder architecture from scratch (no `nn.Transformer`)
- Train on the Multi30k dataset and maximise BLEU score on the test set
- Run 5 ablation experiments (Part 2) to empirically verify key design decisions in the paper
- Log all metrics, attention heatmaps, and translation samples to Weights & Biases

---

## Dataset

**Multi30k** — a multilingual image caption dataset adapted for machine translation.

| Split | Sentences |
|-------|-----------|
| Train | ~29,000   |
| Validation | 1,014 |
| Test  | 1,000     |

- **Language pair:** German (source) → English (target)
- **Source vocabulary:** 18,669 tokens (min_freq=1)
- **Target vocabulary:** 9,797 tokens (min_freq=1)
- Loaded via the HuggingFace `datasets` library (`bentrevett/multi30k`)

---

## Paper: Attention Is All You Need (Vaswani et al., 2017)

The paper introduced the Transformer — a sequence-to-sequence model that relies entirely on attention mechanisms, discarding recurrence and convolutions. Key contributions:

- **Multi-Head Self-Attention** allows the model to jointly attend to information from different representation subspaces at different positions
- **Scaled Dot-Product Attention** with 1/√dk scaling prevents softmax saturation for large dk
- **Sinusoidal Positional Encoding** injects position information without learnable parameters, enabling extrapolation to unseen sequence lengths
- **Noam Learning Rate Schedule** — linear warmup followed by inverse square-root decay — stabilises training of self-attention layers
- **Label Smoothing (ε=0.1)** acts as a regulariser, preventing overconfident output distributions
- **Weight Tying** shares the target embedding matrix with the output projection layer, reducing parameters and improving generalisation (Section 3.4)

---

## Architecture

### Model Configuration

| Hyperparameter | Value |
|---|---|
| d_model | 512 |
| Encoder/Decoder layers (N) | 3 |
| Attention heads | 8 |
| d_k = d_v | 64 (= d_model / heads) |
| Feed-Forward dim (d_ff) | 2048 |
| Dropout | 0.1 |
| Max sequence length | 256 |
| Label smoothing (ε) | 0.1 |
| Warmup steps | 4000 |
| Batch size | 256 |
| Epochs | 40 |

### Components

**Scaled Dot-Product Attention:**
```
Attention(Q, K, V) = softmax(Q·Kᵀ / √dk) · V
```

**Multi-Head Attention:**
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_O
where head_i = Attention(Q·W_Qi, K·W_Ki, V·W_Vi)
```

**Positional Encoding (Sinusoidal):**
```
PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
```

**Noam Learning Rate Schedule:**
```
lr = d_model^(-0.5) × min(step^(-0.5), step × warmup_steps^(-1.5))
```

**Label Smoothing Loss:**
```
q(correct token)    = 1 - ε + ε/V
q(each other token) = ε/V
```

**Weight Tying:**  
The target embedding weight matrix is shared with the output linear projection layer, reducing the parameter count by ~5M and improving generalisation (as described in Section 3.4 of the paper).

### Encoder Block (×3)
```
Input → Multi-Head Self-Attention → Add & LayerNorm
      → Feed-Forward Network      → Add & LayerNorm
```

### Decoder Block (×3)
```
Input → Masked Multi-Head Self-Attention → Add & LayerNorm
      → Multi-Head Cross-Attention        → Add & LayerNorm
      → Feed-Forward Network              → Add & LayerNorm
```

---

## Preprocessing & Tokenization

- **Tokenizer:** spaCy (`de_core_news_sm` for German, `en_core_web_sm` for English)
- **Lowercased:** Yes — all tokens converted to lowercase
- **Min frequency:** 1 — all words that appear at least once in training are included in the vocabulary
- **Special tokens:**

| Token | Index |
|---|---|
| `<unk>` | 0 |
| `<pad>` | 1 |
| `<sos>` | 2 |
| `<eos>` | 3 |

- **Vocabulary files:** `vocab_src.json` and `vocab_tgt.json` are pre-generated from the training split and loaded at inference time — the dataset is never reloaded during inference.

---

## Model Performance

| Metric | Score |
|---|---|
| Best Validation BLEU | 38.93 |
| Test BLEU (local, sacrebleu tokenize=none) | 38.28 |
| Test BLEU (Gradescope autograder) | 38.32 |

BLEU is computed using [sacrebleu](https://github.com/mjpost/sacrebleu) at corpus level. Decoding uses **greedy decoding** (argmax at each step, max length 100 tokens).

---

## Folder Structure

```text
da6401_assignment_3/
├── model.py                      # Full Transformer architecture + infer() method
├── train.py                      # Training loop, Noam scheduler, greedy decoding, W&B logging
├── dataset.py                    # Multi30k loading, spaCy tokenization, vocabulary building
├── lr_scheduler.py               # Noam LR scheduler implementation
├── save_vocab.py                 # One-time script to generate vocab_src.json & vocab_tgt.json
├── hyperparameter_search.py      # Automated HP search over d_model, N, dropout, etc.
├── requirements.txt              # Python dependencies
├── vocab_src.json                # German vocabulary (18,669 tokens)
├── vocab_tgt.json                # English vocabulary (9,797 tokens)
├── training_summary.json         # Best run hyperparameters and results
└── hp_search/
    ├── best_config.json          # Best hyperparameter configuration found
    └── hp_results.json           # All HP search run results
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

### 2. Generate vocabulary files (run once)

```bash
python save_vocab.py
```

This reads the Multi30k training split and saves `vocab_src.json` and `vocab_tgt.json`.

### 3. Train the model

```bash
python train.py
```

Training logs metrics to W&B automatically. The best checkpoint is saved to `checkpoints/`.

### 4. Run inference

```python
from model import Transformer

model = Transformer()          # auto-downloads checkpoint from Google Drive
model.eval()

translation = model.infer("Ein Hund läuft schnell über die lange Brücke im Park.")
print(translation)
# → "a dog runs fast over the long bridge in the park ."
```

No dataset loading or vocab building needed — `vocab_src.json` and `vocab_tgt.json` are loaded automatically from the same directory as `model.py`.


## Dependencies

| Package | Purpose |
|---|---|
| `torch` | Model, training, GPU acceleration |
| `spacy` | German and English tokenization |
| `datasets` | Loading Multi30k from HuggingFace |
| `sacrebleu` | Corpus-level BLEU evaluation |
| `wandb` | Experiment tracking and visualisation |
| `gdown` | Auto-download model checkpoint from Google Drive |
| `tqdm` | Training progress bars |
| `numpy`, `matplotlib` | Attention heatmap generation |
