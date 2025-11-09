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
        out = self.model(input_ids, attention_mask=attention_mask)
        last_hidden = self.model(input_ids, attention_mask=attention_mask).last_hidden_state
        cls_like = last_hidden[:, 0]              # <s> токен — репрезентація речення
        return self.dropout(cls_like)             # [B, hidden]


class BaselineClassifier(nn.Module):
    def __init__(self, plm_name: str = "xlm-roberta-base", use_rating: bool = False):
        super().__init__()
        cfg = AutoConfig.from_pretrained(plm_name)
        self.encoder = NewsEncoder(plm_name)
        self.use_rating = use_rating

        # якщо використовуємо rating — додаємо ще один вхідний вимір
        in_dim = cfg.hidden_size + (1 if use_rating else 0)

        self.head = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)                    # логіт класу "fake"
        )

    def forward(self, input_ids, attention_mask, rating: torch.Tensor = None):
        """
        input_ids: [B, L]
        attention_mask: [B, L]
        rating: [B] або [B, 1] якщо use_rating=True
        """
        text_vec = self.encoder(input_ids, attention_mask)  # [B, H]
        if self.use_rating:
            if rating is None:
                raise ValueError(f"There in {self.__class__.__name__} `use_rating=True` but `rating=None`")
            if rating.dim() == 1:
                rating = rating.unsqueeze(1)  # [B] -> [B,1]
            x = torch.cat([text_vec, rating], dim=1)  # [B, H+1]
        else:
            x = text_vec  # [B, H]

        logits = self.head(x).squeeze(1)  # [B]
        # x = self.encoder(input_ids, attention_mask)
        # logits = self.head(x).squeeze(1)         # [B]
        return logits
