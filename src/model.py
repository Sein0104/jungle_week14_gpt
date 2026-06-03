# -*- coding: utf-8 -*-
"""GPT 모델 구성 요소 구현."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .attention import MultiHeadAttention
    from .embeddings import InputEmbedding
except ImportError:
    from attention import MultiHeadAttention
    from embeddings import InputEmbedding


class LayerNorm(nn.Module):
    """마지막 차원 기준 Layer Normalization."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        # 모집단 분산 (unbiased=False) — GPT-2 구현과 일관
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta


class GELU(nn.Module):
    """GPT FeedForward에서 사용하는 GELU 활성화 함수 (tanh 근사식)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # GPT-2 에서 쓰는 tanh approximation
        return (
            0.5
            * x
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi)
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


class FeedForward(nn.Module):
    """Transformer FFN: Linear -> GELU -> Linear -> Dropout."""

    def __init__(self, d_model: int, dropout: float = 0.1, mult: int = 4):
        super().__init__()
        hidden_dim = mult * d_model
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.act = GELU()
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    GPT block (Pre-LayerNorm 방식 — GPT-2 와 동일):
        x -> LayerNorm -> Causal Self-Attention -> + x (residual)
          -> LayerNorm -> FeedForward            -> + x (residual)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        drop_rate: float = 0.1,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            drop_rate=drop_rate,
            qkv_bias=qkv_bias,
        )
        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout=drop_rate)

    def forward(self, x: torch.Tensor, causal_mask: bool = True) -> torch.Tensor:
        # 1) Pre-LN attention sub-block with residual
        x = x + self.attn(self.ln1(x), causal_mask=causal_mask)
        # 2) Pre-LN FFN sub-block with residual
        x = x + self.ffn(self.ln2(x))
        return x


class GPTModel(nn.Module):
    """InputEmbedding -> TransformerBlock N개 -> LayerNorm -> LM head."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        vocab_size = config["vocab_size"]
        context_length = config["context_length"]
        emb_dim = config["emb_dim"]
        n_heads = config["n_heads"]
        n_layers = config["n_layers"]
        drop_rate = config.get("drop_rate", 0.1)
        qkv_bias = config.get("qkv_bias", False)

        self.input_embedding = InputEmbedding(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            context_length=context_length,
            drop_rate=drop_rate,
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=emb_dim,
                    n_heads=n_heads,
                    drop_rate=drop_rate,
                    qkv_bias=qkv_bias,
                )
                for _ in range(n_layers)
            ]
        )

        self.final_ln = LayerNorm(emb_dim)
        # GPT-2 관례: LM head 는 bias 없음
        self.lm_head = nn.Linear(emb_dim, vocab_size, bias=False)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ):
        """
        Args:
            idx: (batch_size, seq_len) token IDs
            targets: (batch_size, seq_len) target token IDs (optional)

        Returns:
            targets 가 None 이면 logits (B, T, vocab_size)
            targets 가 있으면 (loss, logits)
        """
        x = self.input_embedding(idx)              # (B, T, C)
        for block in self.blocks:
            x = block(x, causal_mask=True)         # (B, T, C)
        x = self.final_ln(x)                       # (B, T, C)
        logits = self.lm_head(x)                   # (B, T, vocab_size)

        if targets is None:
            return logits

        # cross entropy: (B*T, V) vs (B*T,)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )
        return loss, logits


def generate_text_simple(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
) -> torch.Tensor:
    """
    greedy 방식으로 max_new_tokens 만큼 다음 토큰을 이어 붙인다.

    Args:
        model: GPTModel
        idx: (batch_size, seq_len) 시작 토큰
        max_new_tokens: 생성할 토큰 수
        context_size: 모델이 한 번에 볼 수 있는 최대 context 길이

    Returns:
        (batch_size, seq_len + max_new_tokens)
    """
    model.eval()
    for _ in range(max_new_tokens):
        # context window 잘라내기
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            out = model(idx_cond, targets=None)
        # GPTModel.forward 가 logits 만 반환하는 경로
        logits = out if not isinstance(out, tuple) else out[1]

        # 마지막 시점의 logits 만 사용
        next_logits = logits[:, -1, :]                                  # (B, V)
        next_token = torch.argmax(next_logits, dim=-1, keepdim=True)    # (B, 1)

        idx = torch.cat([idx, next_token], dim=1)
    return idx
