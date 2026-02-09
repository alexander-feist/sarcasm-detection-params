import torch
from torch import nn
from transformers import BertModel

from src.model.head_constructor import ClassificationHeadConstructor


class Classifier(nn.Module):
    """Classifier model."""

    def __init__(self, pretrained_model_name: str, head: ClassificationHeadConstructor) -> None:
        """Initialize the classifier model."""
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.head = head(self.bert.config.hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run forward pass."""
        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            return_dict=True,
        )
        return self.head(bert_out)

    @property
    def num_params(self) -> int:
        """Number of trainable parameters in the model."""
        return sum(
            int(torch.prod(torch.tensor(p.size()))) for p in self.parameters() if p.requires_grad
        )
