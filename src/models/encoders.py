from torch import nn
from transformers import AutoModel


class NewsEncoder(nn.Module):
    def __init__(self, plm_name: str = "xlm-roberta-base", dropout_p: float = 0.1):
        super().__init__()
        self.model = AutoModel.from_pretrained(plm_name)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids, attention_mask=attention_mask)
        last_hidden = out.last_hidden_state          # [B, L, H]
        cls_like = last_hidden[:, 0]                 # [B, H]
        return self.dropout(cls_like)
