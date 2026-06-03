# -*- coding: utf-8 -*-
"""Multi-Head Self-Attention 구현."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    GPT의 causal self-attention.

    - Q/K/V projection
    - head 분리: (B, T, C) -> (B, n_heads, T, head_dim)
    - attention score = QK^T / sqrt(head_dim)
    - causal mask로 미래 토큰 가리기
    - attention weight와 V를 곱한 뒤 head를 다시 합치기
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        drop_rate: float = 0.1,
        qkv_bias: bool = False,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Q, K, V projection
        self.W_q = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.W_k = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.W_v = nn.Linear(d_model, d_model, bias=qkv_bias)

        # 출력 projection
        self.out_proj = nn.Linear(d_model, d_model)

        # dropout
        self.attn_dropout = nn.Dropout(drop_rate)
        self.resid_dropout = nn.Dropout(drop_rate)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: bool = True,
        return_attention_weights: bool = False,
    ):
        """
        Args:
            x: (batch_size, seq_len, d_model)
            causal_mask: True이면 미래 위치를 볼 수 없게 mask 처리
            return_attention_weights: True이면 attention weight도 함께 반환
        """
        B, T, C = x.shape

        # 1) Q, K, V 계산: (B, T, C)
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # 2) head 분리: (B, T, C) -> (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # 3) attention score: (B, n_heads, T, T)
        attn_scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)

        # 4) causal mask: 미래 위치를 -inf 로
        if causal_mask:
            mask = torch.triu(
                torch.ones(T, T, dtype=torch.bool, device=x.device),
                diagonal=1,
            )
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        # 5) softmax + dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # 6) attention weight × V: (B, n_heads, T, head_dim)
        context = attn_weights @ v

        # 7) head 합치기: (B, T, C)
        context = context.transpose(1, 2).contiguous().view(B, T, C)

        # 8) 출력 projection + residual dropout
        out = self.out_proj(context)
        out = self.resid_dropout(out)

        if return_attention_weights:
            return out, attn_weights
        return out
