"""
model.py  Transformer Architecture
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────┐
  │  scaled_dot_product_attention(Q, K, V, mask) → (out, weights)  │
  │  MultiHeadAttention.forward(q, k, v, mask)   → Tensor          │
  │  PositionalEncoding.forward(x)               → Tensor          │
  │  make_src_mask(src, pad_idx)                 → BoolTensor      │
  │  make_tgt_mask(tgt, pad_idx)                 → BoolTensor      │
  │  Transformer.encode(src, src_mask)           → Tensor          │
  │  Transformer.decode(memory,src_m,tgt,tgt_m)  → Tensor          │
  └─────────────────────────────────────────────────────────────────┘

Design notes:
  - Post-LayerNorm (original paper §3) inside each sub-layer, plus a
    final LayerNorm at the top of each stack.
  - use_scaling  flag propagated to scaled_dot_product_attention for
    experiment 2.2 (ablation of 1/√dk).
  - pe_type      flag on Transformer selects sinusoidal vs learned PE
    for experiment 2.4.
  - MultiHeadAttention stores self.attn_weights for attention heatmap
    visualisation in experiment 2.3.
"""

import json
import math
import copy
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


#  STANDALONE ATTENTION FUNCTION

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    use_scaling: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention.

        Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V

    Args:
        Q           : Query tensor,  shape (..., seq_q, d_k)
        K           : Key tensor,    shape (..., seq_k, d_k)
        V           : Value tensor,  shape (..., seq_k, d_v)
        mask        : Optional Boolean mask broadcastable to
                      (..., seq_q, seq_k).
                      True  → position is MASKED OUT (set to -inf).
        use_scaling : If True, divide scores by √dₖ (default True).
                      Set False for experiment 2.2 ablation.

    Returns:
        output  : Attended output,   shape (..., seq_q, d_v)
        attn_w  : Attention weights, shape (..., seq_q, seq_k)
    """
    d_k = Q.size(-1)

    # Raw attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1))   # (..., seq_q, seq_k)
    if use_scaling:
        scores = scores / math.sqrt(d_k)

    # Apply mask: positions where mask is True become -inf
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    # Softmax over key dimension; replace NaN (all-masked rows) with 0
    attn_weights = F.softmax(scores, dim=-1)
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    output = torch.matmul(attn_weights, V)           # (..., seq_q, d_v)
    return output, attn_weights


#  MASK HELPERS

def make_src_mask(
    src: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Padding mask for the encoder (source sequence).

    Args:
        src     : [batch, src_len]
        pad_idx : index of <pad> token (default 1)

    Returns:
        BoolTensor [batch, 1, 1, src_len]
        True  → PAD (masked out)   False → real token
    """
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(
    tgt: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Combined padding + causal (look-ahead) mask for the decoder.

    Args:
        tgt     : [batch, tgt_len]
        pad_idx : index of <pad> token (default 1)

    Returns:
        BoolTensor [batch, 1, tgt_len, tgt_len]
        True → masked out (PAD token or future position)
    """
    batch_size, tgt_len = tgt.shape

    # Padding mask: True where tgt token is <pad>
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)          # [B,1,1,T]

    # Causal mask: upper-triangular matrix, True at future positions
    causal_mask = torch.triu(
        torch.ones(tgt_len, tgt_len, device=tgt.device, dtype=torch.bool),
        diagonal=1,
    )                                                               # [T, T]

    # Broadcast-combine: [B,1,1,T] | [T,T] → [B,1,T,T]
    return pad_mask | causal_mask


#  MULTI-HEAD ATTENTION

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention as in "Attention Is All You Need", §3.2.2.

    Does NOT use torch.nn.MultiheadAttention.
    Stores self.attn_weights after every forward pass for visualisation.

    Args:
        d_model    : Total model dimensionality (must be divisible by num_heads).
        num_heads  : Number of parallel attention heads.
        dropout    : Dropout applied to the attention output before W_O.
        use_scaling: Passed through to scaled_dot_product_attention.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        use_scaling: bool = True,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model    = d_model
        self.num_heads  = num_heads
        self.d_k        = d_model // num_heads
        self.use_scaling = use_scaling

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout)

        # Stored after every forward  used for attention heatmaps (exp 2.3)
        self.attn_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query : [batch, seq_q, d_model]
            key   : [batch, seq_k, d_model]
            value : [batch, seq_k, d_model]
            mask  : BoolTensor broadcastable to [batch, heads, seq_q, seq_k]

        Returns:
            output : [batch, seq_q, d_model]
        """
        B = query.size(0)

        # Linear projections → split into heads
        def project_and_split(linear, x):
            # [B, seq, d_model] → [B, heads, seq, d_k]
            return linear(x).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)

        Q = project_and_split(self.W_q, query)   # [B, H, seq_q, d_k]
        K = project_and_split(self.W_k, key)     # [B, H, seq_k, d_k]
        V = project_and_split(self.W_v, value)   # [B, H, seq_k, d_k]

        # Scaled dot-product attention
        attn_out, attn_weights = scaled_dot_product_attention(
            Q, K, V, mask=mask, use_scaling=self.use_scaling
        )                                        # [B, H, seq_q, d_k], [B, H, seq_q, seq_k]

        # Store weights for later visualisation (detached)
        self.attn_weights = attn_weights.detach()

        # Concatenate heads
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, -1, self.d_model)

        # Final projection + dropout
        return self.dropout(self.W_o(attn_out))  # [B, seq_q, d_model]


#  POSITIONAL ENCODING  sinusoidal (registered buffer, not trainable)

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding as in "Attention Is All You Need", §3.5.

    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    The PE tensor is stored as a non-trainable buffer.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe       = torch.zeros(max_len, d_model)                        # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model)
        )                                                               # [d_model/2]

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Shape: [1, max_len, d_model] so it broadcasts over batch
        pe = pe.unsqueeze(0)

        # Register as buffer: saved in state_dict, not a trainable parameter
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [batch, seq_len, d_model]
        Returns:
            [batch, seq_len, d_model]
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


#  POSITIONAL ENCODING  learned (nn.Embedding, for experiment 2.4)

class LearnedPositionalEncoding(nn.Module):
    """
    Learned Positional Encoding via nn.Embedding.
    Used in experiment 2.4 to compare against sinusoidal encoding.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout   = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [batch, seq_len, d_model]
        Returns:
            [batch, seq_len, d_model]
        """
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)  # [1, seq_len]
        x = x + self.embedding(positions)
        return self.dropout(x)


#  FEED-FORWARD NETWORK

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network, §3.3:
        FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout  = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [batch, seq_len, d_model]
        Returns:
            [batch, seq_len, d_model]
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


#  ENCODER LAYER

class EncoderLayer(nn.Module):
    """
    Single Transformer encoder sub-layer (Post-LayerNorm, original paper):
        x → Self-Attn → Dropout → Add & Norm → FFN → Dropout → Add & Norm
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_scaling: bool = True,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, use_scaling)
        self.ff        = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(p=dropout)
        self.d_model   = d_model

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x        : [batch, src_len, d_model]
            src_mask : [batch, 1, 1, src_len]
        Returns:
            [batch, src_len, d_model]
        """
        # Self-attention sublayer
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))
        # Feed-forward sublayer
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


#  DECODER LAYER

class DecoderLayer(nn.Module):
    """
    Single Transformer decoder sub-layer (Post-LayerNorm):
        x → Masked Self-Attn → Add & Norm
          → Cross-Attn(memory) → Add & Norm
          → FFN → Add & Norm
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_scaling: bool = True,
    ) -> None:
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout, use_scaling)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout, use_scaling)
        self.ff         = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(p=dropout)
        self.d_model    = d_model

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : [batch, tgt_len, d_model]
            memory   : encoder output, [batch, src_len, d_model]
            src_mask : [batch, 1, 1, src_len]
            tgt_mask : [batch, 1, tgt_len, tgt_len]
        Returns:
            [batch, tgt_len, d_model]
        """
        # Masked self-attention
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        # Cross-attention: queries from decoder, keys/values from encoder
        x = self.norm2(x + self.dropout(self.cross_attn(x, memory, memory, src_mask)))
        # Feed-forward
        x = self.norm3(x + self.dropout(self.ff(x)))
        return x


#  ENCODER & DECODER STACKS

class Encoder(nn.Module):
    """Stack of N identical EncoderLayer modules with final LayerNorm."""

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x    : [batch, src_len, d_model]
            mask : [batch, 1, 1, src_len]
        Returns:
            [batch, src_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """Stack of N identical DecoderLayer modules with final LayerNorm."""

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.d_model)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : [batch, tgt_len, d_model]
            memory   : [batch, src_len, d_model]
            src_mask : [batch, 1, 1, src_len]
            tgt_mask : [batch, 1, tgt_len, tgt_len]
        Returns:
            [batch, tgt_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


#  FULL TRANSFORMER

class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for sequence-to-sequence tasks.

    Args:
        src_vocab_size : Source vocabulary size.
        tgt_vocab_size : Target vocabulary size.
        d_model        : Model dimensionality (default 512).
        N              : Number of encoder/decoder layers (default 6).
        num_heads      : Number of attention heads (default 8).
        d_ff           : FFN inner dimensionality (default 2048).
        dropout        : Dropout probability (default 0.1).
        pe_type        : 'sinusoidal' (default) or 'learned'  exp 2.4.
        use_scaling    : Whether to use 1/√dk scaling  exp 2.2.
    """

    def __init__(
        self,
        src_vocab_size: int   = 7853,
        tgt_vocab_size: int   = 5893,
        d_model:    int   = 512,
        N:          int   = 3,
        num_heads:  int   = 8,
        d_ff:       int   = 2048,
        dropout:    float = 0.1,
        pe_type:    str   = "sinusoidal",
        use_scaling: bool = True,
        load_pretrained: bool = True,
    ) -> None:
        super().__init__()

        self.d_model   = d_model
        self.pe_type   = pe_type
        self.use_scaling = use_scaling

        # Token embeddings
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)

        # Positional encodings
        def _make_pe():
            if pe_type == "sinusoidal":
                return PositionalEncoding(d_model, dropout)
            return LearnedPositionalEncoding(d_model, dropout)

        self.src_pe = _make_pe()
        self.tgt_pe = _make_pe()

        # Encoder / Decoder stacks
        enc_layer = EncoderLayer(d_model, num_heads, d_ff, dropout, use_scaling)
        dec_layer = DecoderLayer(d_model, num_heads, d_ff, dropout, use_scaling)

        self.encoder = Encoder(enc_layer, N)
        self.decoder = Decoder(dec_layer, N)

        # Final linear projection to vocabulary
        self.projection = nn.Linear(d_model, tgt_vocab_size)

        # Store config for checkpoint reconstruction
        self._model_config = {
            "src_vocab_size": src_vocab_size,
            "tgt_vocab_size": tgt_vocab_size,
            "d_model":        d_model,
            "N":              N,
            "num_heads":      num_heads,
            "d_ff":           d_ff,
            "dropout":        dropout,
            "pe_type":        pe_type,
            "use_scaling":    use_scaling,
        }

        self._init_weights()

        if load_pretrained:
            self._load_resources()

    def _load_resources(self) -> None:
        """Load spaCy tokenizer, vocab JSON files, and pretrained weights from Google Drive."""
        import spacy
        try:
            self._spacy_de = spacy.load("de_core_news_sm")
        except OSError:
            from spacy.cli import download as spacy_download
            spacy_download("de_core_news_sm")
            self._spacy_de = spacy.load("de_core_news_sm")

        _dir = os.path.dirname(os.path.abspath(__file__))

        with open(os.path.join(_dir, "vocab_src.json")) as f:
            _src = json.load(f)
        with open(os.path.join(_dir, "vocab_tgt.json")) as f:
            _tgt = json.load(f)
        self._src_stoi: dict = _src["stoi"]
        self._src_itos: list = _src["itos"]
        self._tgt_stoi: dict = _tgt["stoi"]
        self._tgt_itos: list = _tgt["itos"]

        _GDRIVE_FILE_ID = "10SJOX24y8yv1oMWxfDBjNsflCjfyKkXs"
        _ckpt_path = os.path.join(_dir, "checkpoints", "best_checkpoint.pt")
        if not os.path.exists(_ckpt_path):
            os.makedirs(os.path.join(_dir, "checkpoints"), exist_ok=True)
            import gdown
            gdown.download(
                f"https://drive.google.com/uc?id={_GDRIVE_FILE_ID}",
                _ckpt_path,
                quiet=False,
            )
        _ckpt = torch.load(_ckpt_path, map_location="cpu")
        self.load_state_dict(_ckpt["model_state_dict"])

    def _init_weights(self) -> None:
        """Xavier uniform initialisation for all weight matrices."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ── AUTOGRADER HOOKS ── keep these signatures exactly ─────────────

    def encode(
        self,
        src:      torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            src      : [batch, src_len]
            src_mask : [batch, 1, 1, src_len]
        Returns:
            memory   : [batch, src_len, d_model]
        """
        # Scale embeddings by √d_model (§3.4) then add positional encoding
        x = self.src_pe(self.src_embed(src) * math.sqrt(self.d_model))
        return self.encoder(x, src_mask)

    def decode(
        self,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt:      torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            memory   : [batch, src_len, d_model]
            src_mask : [batch, 1, 1, src_len]
            tgt      : [batch, tgt_len]
            tgt_mask : [batch, 1, tgt_len, tgt_len]
        Returns:
            logits   : [batch, tgt_len, tgt_vocab_size]
        """
        x = self.tgt_pe(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        x = self.decoder(x, memory, src_mask, tgt_mask)
        return self.projection(x)

    def forward(
        self,
        src:      torch.Tensor,
        tgt:      torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full encoder-decoder forward pass.

        Args:
            src      : [batch, src_len]
            tgt      : [batch, tgt_len]
            src_mask : [batch, 1, 1, src_len]
            tgt_mask : [batch, 1, tgt_len, tgt_len]
        Returns:
            logits   : [batch, tgt_len, tgt_vocab_size]
        """
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    # ── Helpers for Part 2 visualisations 

    def get_last_encoder_attn(self) -> Optional[torch.Tensor]:
        """
        Return the self-attention weights from the final encoder layer.
        Shape: [batch, num_heads, src_len, src_len]
        Must be called right after encode() while weights are still stored.
        """
        return self.encoder.layers[-1].self_attn.attn_weights

    def infer(self, german_sentence: str) -> str:
        """
        End-to-end German → English translation.

        Accepts a raw German string, tokenizes it with spaCy, runs the
        Transformer forward pass with autoregressive greedy decoding, and
        returns the translated English sentence.

        Args:
            german_sentence : Raw German input string.

        Returns:
            Translated English string.
        """
        device = next(self.parameters()).device

        # Tokenise German sentence
        tokens = [tok.text.lower() for tok in self._spacy_de.tokenizer(german_sentence)]

        # Numericalize: <sos>=2, <eos>=3, <unk>=0
        src_ids = [2] + [self._src_stoi.get(t, 0) for t in tokens] + [3]
        src = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)
        src_mask = make_src_mask(src)

        # Autoregressive greedy decode
        self.eval()
        with torch.no_grad():
            memory = self.encode(src, src_mask)
            ys = torch.tensor([[2]], device=device)   # start with <sos>

            for _ in range(100):
                tgt_mask = make_tgt_mask(ys)
                logits   = self.decode(memory, src_mask, ys, tgt_mask)
                next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                ys = torch.cat([ys, next_tok], dim=1)
                if next_tok.item() == 3:              # <eos>
                    break

        # Detokenise  skip <pad>=1, <sos>=2, <eos>=3
        skip = {1, 2, 3}
        out_tokens = [self._tgt_itos[i] for i in ys.squeeze(0).tolist() if i not in skip]
        return " ".join(out_tokens)
