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
        last_hidden = out.last_hidden_state          # [B, L, H]
        cls_like = last_hidden[:, 0]                 # [B, H]
        return self.dropout(cls_like)


class BaselineClassifier(nn.Module):
    def __init__(
        self,
        plm_name: str = "xlm-roberta-base",
        use_rating: bool = False,
        use_author_prior: bool = False
    ):
        super().__init__()
        cfg = AutoConfig.from_pretrained(plm_name)
        self.encoder = NewsEncoder(plm_name)
        self.use_rating = use_rating
        self.use_author_prior = use_author_prior

        extra_dims = 0
        if use_rating:
            extra_dims += 1
        if use_author_prior:
            extra_dims += 1

        in_dim = cfg.hidden_size + extra_dims

        self.head = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask,
                rating: torch.Tensor | None = None,
                author_prior: torch.Tensor | None = None):
        text_vec = self.encoder(input_ids, attention_mask)   # [B, H]
        feats = [text_vec]

        if self.use_rating:
            if rating is None:
                raise ValueError("use_rating=True, але rating=None")
            if rating.dim() == 1:
                rating = rating.unsqueeze(1)
            feats.append(rating)

        if self.use_author_prior:
            if author_prior is None:
                raise ValueError("use_author_prior=True, але author_prior=None")
            if author_prior.dim() == 1:
                author_prior = author_prior.unsqueeze(1)
            feats.append(author_prior)

        x = torch.cat(feats, dim=1)                          # [B, H + extra]
        logits = self.head(x).squeeze(1)                     # [B]
        return logits
