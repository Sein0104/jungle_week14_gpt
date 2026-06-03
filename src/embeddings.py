# -*- coding: utf-8 -*-
"""토큰 임베딩 + 위치 임베딩."""

import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    """
    token ID를 Transformer 입력 벡터로 바꿉니다.

    구조:
    - token embedding: nn.Embedding(vocab_size, emb_dim)
    - position embedding: nn.Embedding(context_length, emb_dim)
    - token embedding + position embedding
    - dropout
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        context_length: int,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.context_length = context_length

        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        self.position_embedding = nn.Embedding(context_length, emb_dim)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len) token IDs

        Returns:
            (batch_size, seq_len, emb_dim)
        """
        _, seq_len = x.shape

        # 0, 1, 2, ... seq_len-1
        positions = torch.arange(seq_len, device=x.device)

        token_emb = self.token_embedding(x)              # (B, T, C)
        pos_emb = self.position_embedding(positions)     # (T, C)

        # broadcasting: (B, T, C) + (T, C) -> (B, T, C)
        out = token_emb + pos_emb
        out = self.dropout(out)
        return out
