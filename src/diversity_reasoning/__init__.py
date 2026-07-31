"""Diversity-vs-reasoning research pipeline."""

from .metrics import QOrder, pseudo_logdet, vendi_score

__all__ = ["QOrder", "pseudo_logdet", "vendi_score"]
__version__ = "0.1.0"
