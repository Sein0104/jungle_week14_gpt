# -*- coding: utf-8 -*-
"""토큰 임베딩 + 위치 임베딩 과제 템플릿."""

import torch
import torch.nn as nn

"""
nnModule을 상속한 class InputEmbedding을 만듭니다.
nnModule은 파이토치에서 모든 딥러닝 레이어의 부모 클래스가 됩니다.
역전파, 가중치 관리 GPU 이동 같은 기능이 이 클래스에 있습니다.
"""
class InputEmbedding(nn.Module):
    """
    token ID를 Transformer 입력 벡터로 바꿉니다.

    구현할 구조:
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
        #부모 클래스의 __init__을 호출합니다.
        super().__init__()
        self.emb_dim = emb_dim
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.drop_rate = drop_rate
        # TODO: token_embedding, position_embedding, dropout을 정의하세요.
        self.token_embedding = nn.Embedding(self.vocab_size, self.emb_dim)
        self.position_embedding = nn.Embedding(self.context_length, self.emb_dim)
        self.dropout = nn.Dropout(self.drop_rate)
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        TODO: token embedding과 position embedding을 더한 뒤 dropout을 적용합니다.
        
        두 임베딩을 더하면, 벡터가 단어의 의미뿐 아니라 위치의 의미까지 갖게 됩니다.
        
        ?? 왜 Drop Out을 해야할까요? 
        특정 뉴런에만 의존하게 되면 훈련 데이터에는 완벽한데, 새로운 데이터가 들어오면 맞추지 못 합니다. 이게 (훈련 데이터) 과적합입니다.
        학습을 할 때마다 한 뉴런에만 의존하게 되면, 그 뉴런이 잘 처리하는 패턴으로만 처리하게 됩니다.
        그룹에서 의사 결정의 예시에 빗대봅시다. 어떤 문제가 들어 왔을 때 한 사람만 처리를 하면, 모든 문제를 그 방향으로 보게 될 것입니다.
        따라서 가끔은 그 사람을 배제하고, 다른 사람들이 문제를 풀다 보면, 그 사람만의 생각이나 특화된 방향 또한 드러날 것입니다.
        이로 인해 균형잡힌 의사결정을 하고, 다양한 결과를 낼 수 있을 것입니다. Drop Out도 비슷한 맥락에서 과적합을 예방합니다.
        
        

        Args:
            x: (batch_size, seq_len) token IDs

        Returns:
            (batch_size, seq_len, emb_dim)
        """
        print(x)
        vetctor_sum = self.token_embedding(x) + self.position_embedding(torch.arange(x.shape[1]))
        
        return self.dropout(vetctor_sum)
        