import torch
import torch.nn as nn
from src.models.encoders import NewsEncoder
from src.models.heads import TextHead  # якщо нема — дам нижче імплементацію


class AuthorBayesClassifier(nn.Module):
    """
    Bayesian Logit Fusion with style prior:
    logit_final = logit_text + λ1 * logit_prior_rating + λ2 * logit_style
    """
    def __init__(
        self,
        plm_name="xlm-roberta-base",
        d_text=768,
        lambda_prior: float = 1.0,
        lambda_style: float = 1.0,
    ):
        super().__init__()
        self.encoder = NewsEncoder(plm_name=plm_name)
        self.text_head = TextHead(d_text)
        self.lambda_prior = lambda_prior
        self.lambda_style = lambda_style

    def forward(self, ids, attn, rating, author_prior=None, style_score=None):
        """
        rating ~ P(real | author)
        style_score ~ 'наскільки стиль типовий для правдивих текстів автора' в [0,1]
        """
        text_vec = self.encoder(ids, attn)            # [B, d]
        logit_text = self.text_head(text_vec).view(-1)  # [B]

        eps = 1e-6

        # --- пріор по рейтингу автора ---
        pi_fake = (1.0 - rating).clamp(eps, 1.0 - eps)
        logit_prior = torch.log(pi_fake / (1.0 - pi_fake))  # [B]

        # --- стилістичний пріор ---
        if style_score is not None:
            # style_score ~ наскільки це схоже на правдивий стиль => пріор на fake = 1 - style_score
            pi_style_fake = (1.0 - style_score).clamp(eps, 1.0 - eps)
            logit_style = torch.log(pi_style_fake / (1.0 - pi_style_fake))
        else:
            logit_style = 0.0

        logits_final = logit_text \
                       + self.lambda_prior * logit_prior \
                       + self.lambda_style * logit_style

        return logits_final
