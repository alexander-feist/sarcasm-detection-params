from abc import ABC, abstractmethod

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions


class ClassificationHead(ABC, nn.Module):
    """Abstract base for classification heads."""

    def __init__(self, hidden_size: int, dropout_p: float = 0.1) -> None:
        """Initialize the head."""
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.num_classes = 2

    @abstractmethod
    def forward(
        self, encoder_outputs: BaseModelOutputWithPoolingAndCrossAttentions
    ) -> torch.Tensor:
        """Forward through the head."""
        ...


class BertMeanPoolLinear(ClassificationHead):
    """BERT mean pooling head."""

    def __init__(self, hidden_size: int, dropout_p: float = 0.1) -> None:
        """Initialize the head."""
        super().__init__(hidden_size, dropout_p)

        self.dropout = nn.Dropout(self.dropout_p)
        self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def forward(
        self, encoder_outputs: BaseModelOutputWithPoolingAndCrossAttentions
    ) -> torch.Tensor:
        """Forward through the head."""
        pooled = encoder_outputs.pooler_output
        return self.fc(self.dropout(pooled))


class BertCLSMLP(ClassificationHead):
    """BERT CLS head with MLP."""

    def __init__(
        self, hidden_size: int, dropout_p: float = 0.1, intermediate_size: int | None = None
    ) -> None:
        """Initialize the head."""
        super().__init__(hidden_size, dropout_p)
        self.intermediate_size = intermediate_size or 4 * self.hidden_size

        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.intermediate_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.intermediate_size, self.hidden_size),
            nn.ReLU(),
        )
        self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def forward(
        self, encoder_outputs: BaseModelOutputWithPoolingAndCrossAttentions
    ) -> torch.Tensor:
        """Forward through the head."""
        pooled = encoder_outputs.pooler_output
        return self.fc(self.mlp(pooled) + pooled)


class BertMultiHeadAttnPool(ClassificationHead):
    """BERT multi-head attention pooling head."""

    def __init__(self, hidden_size: int, dropout_p: float = 0.1, num_heads: int = 8) -> None:
        """Initialize the head."""
        super().__init__(hidden_size, dropout_p)
        self.num_heads = num_heads

        self.query = nn.Parameter(torch.randn(1, 1, self.hidden_size))
        self.attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_heads,
            dropout=self.dropout_p,
            batch_first=True,
        )
        self.dropout = nn.Dropout(self.dropout_p)
        self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def forward(
        self, encoder_outputs: BaseModelOutputWithPoolingAndCrossAttentions
    ) -> torch.Tensor:
        """Forward through the head."""
        token_embeddings = encoder_outputs.last_hidden_state  # (B, S, H)
        batch_size = token_embeddings.size(0)

        # Expand the query for batch
        query = self.query.expand(batch_size, -1, -1)  # (B, 1, H)

        # Attend over all token embeddings
        attn_out, _ = self.attn(query, token_embeddings, token_embeddings)  # (B, 1, H)

        pooled_output = attn_out.squeeze(1)  # (B, H)
        pooled_output = self.dropout(pooled_output)

        return self.fc(pooled_output)


class BertSingleTokenAttention(ClassificationHead):
    """BERT single token attention head."""

    def __init__(self, hidden_size: int, dropout_p: float = 0.1) -> None:
        """Initialize the head."""
        super().__init__(hidden_size, dropout_p)

        self.attn = nn.Linear(self.hidden_size, 1, bias=False)
        self.dropout = nn.Dropout(self.dropout_p)
        self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def forward(
        self, encoder_outputs: BaseModelOutputWithPoolingAndCrossAttentions
    ) -> torch.Tensor:
        """Forward through the head."""
        h = encoder_outputs.last_hidden_state
        attn_scores = self.attn(h)
        attn_weights = torch.softmax(attn_scores, dim=1)
        pooled = torch.sum(attn_weights * h, dim=1)
        return self.fc(self.dropout(pooled))
