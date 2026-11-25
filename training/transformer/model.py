"""Self-built Encoder-only Transformer: no pretrained architecture or weights."""
from __future__ import annotations
import math
import torch
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

    def split_heads(self, tensor):
        b, length, _ = tensor.shape
        return tensor.reshape(b, length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x, attention_mask):
        q = self.split_heads(self.q_proj(x))
        k = self.split_heads(self.k_proj(x))
        v = self.split_heads(self.v_proj(x))
        logits = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # Attention mask = 1 for actual tokens and CLS; 0 for PAD.
        logits = logits.masked_fill(attention_mask[:, None, None, :] == 0, -1e4)
        weights = torch.softmax(logits, dim=-1)
        weights = self.attn_dropout(weights)
        context = weights @ v
        context = context.transpose(1, 2).contiguous()
        b, length, _, _ = context.shape
        return self.out_proj(context.reshape(b, length, -1))


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask):
        x = x + self.dropout(self.attention(self.norm1(x), attention_mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class SentimentTransformer(nn.Module):
    def __init__(self, vocab_size: int, max_len: int = 128, d_model: int = 128,
                 num_heads: int = 4, num_layers: int = 2, d_ff: int = 256,
                 dropout: float = 0.2, num_classes: int = 3):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        # Position embedding (trainable positional encoding) learned from scratch.
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.embed_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, num_classes))

    def forward(self, input_ids, attention_mask):
        batch_size, length = input_ids.shape
        positions = torch.arange(length, device=input_ids.device).unsqueeze(0)
        x = self.embed_dropout(self.token_embedding(input_ids) + self.position_embedding(positions))
        for block in self.blocks:
            x = block(x, attention_mask)
        x = self.final_norm(x)
        return self.head(x[:, 0, :])  # CLS representation; raw logits for CrossEntropyLoss
