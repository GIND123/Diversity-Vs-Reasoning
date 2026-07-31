"""Shared constants for the experiment matrix."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "base.yaml"
DEFAULT_MODELS_CONFIG = ROOT / "configs" / "models.yaml"
DEFAULT_GATE_STAMP = ROOT / ".correctness-gate"

Q_ORDERS = (0.0, 0.1, 0.5, 1.0, 2.0, float("inf"))
POOL_BUDGETS = (4, 8, 16, 32, 64, 128, 256, 512, 1024)
SELECTION_BUDGETS = (2, 3, 4, 8, 16, 32)
ALPHAS = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
