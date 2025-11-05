# NewsEncoder + BaselineClassifier
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class NewsEncoder(nn.Module):
    def __init__(self, plm_name: str = "xlm-roberta-base", dropout_p: float = 0.1):
        super().__init__()
        self.model = AutoModel.from_pretrained(plm_name)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input_ids, attention_mask):
        last_hidden = self.model(input_ids, attention_mask=attention_mask).last_hidden_state
        cls_like = last_hidden[:, 0]              # <s> токен — репрезентація речення
        return self.dropout(cls_like)             # [B, hidden]

class BaselineClassifier(nn.Module):
    def __init__(self, plm_name: str = "xlm-roberta-base"):
        super().__init__()
        cfg = AutoConfig.from_pretrained(plm_name)
        self.encoder = NewsEncoder(plm_name)
        self.head = nn.Sequential(
            nn.Linear(cfg.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)                    # логіт класу "fake"
        )

    def forward(self, input_ids, attention_mask):
        x = self.encoder(input_ids, attention_mask)
        logits = self.head(x).squeeze(1)         # [B]
        return logits
