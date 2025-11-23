import torch
import torch.nn as nn
from src.models.baseline_classifier import NewsEncoder
from src.models.heads import TextHead  # якщо нема — дам нижче імплементацію


class AuthorBayesClassifier(nn.Module):
    """
    Bayesian Logit Fusion:
    final_logit = logit_text + lambda_prior * logit_prior_from_rating
    where:
        logit_prior = log( (1-rating) / rating )
    """
    def __init__(self, plm_name="xlm-roberta-base", d_text=768, lambda_prior=1.0):
        super().__init__()
        self.encoder = NewsEncoder(plm_name=plm_name)
        self.text_head = TextHead(d_text)
        self.lambda_prior = lambda_prior

    def forward(self, ids, attn, rating, author_prior=None):
        # 1) текст → CLS-вектор
        text_vec = self.encoder(ids, attn)            # [B, d_text]

        # 2) текстовий логіт
        logit_text = self.text_head(text_vec).view(-1)  # [B]

        # 3) prior для P(fake|author) = 1 - rating
        eps = 1e-6
        pi_fake = (1.0 - rating).clamp(eps, 1.0 - eps)
        # log-odds від prior
        logit_prior = torch.log(pi_fake / (1.0 - pi_fake))  # [B]

        # 4) Байєсівське комбінування логітів
        final_logits = logit_text + self.lambda_prior * logit_prior

        return final_logits
