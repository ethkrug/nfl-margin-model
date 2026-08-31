"""Lightweight, dependency-free pretty-printing helpers.

These give the pipeline a clean, readable terminal output (section rules,
step markers, metric tables, dataframe previews) without pulling in any
external library. Colors are emitted only when stdout is a real terminal.
"""

from __future__ import annotations

import sys
import pandas as pd

# ANSI styles, disabled automatically when output is redirected to a file/pipe.
_TTY = sys.stdout.isatty()


def _c(code: str) -> str:
    return code if _TTY else ""


BOLD = _c("\033[1m")
DIM = _c("\033[2m")
CYAN = _c("\033[36m")
GREEN = _c("\033[32m")
YELLOW = _c("\033[33m")
BLUE = _c("\033[34m")
RESET = _c("\033[0m")

_WIDTH = 78


def rule(title: str) -> None:
    """Print a titled horizontal rule that opens a section."""
    title = f" {title} "
    pad = _WIDTH - len(title)
    left = pad // 2
    right = pad - left
    print()
    print(f"{BOLD}{CYAN}{'─' * left}{title}{'─' * right}{RESET}")


def step(message: str) -> None:
    """Print a numbered/itemized progress step."""
    print(f"  {GREEN}▸{RESET} {message}")


def info(message: str) -> None:
    """Print a secondary, dimmed informational line."""
    print(f"    {DIM}{message}{RESET}")


def metrics_table(label: str, rmse: float, mae: float, r2: float) -> None:
    """Print an aligned table of regression metrics."""
    print(f"  {BOLD}{BLUE}{label}{RESET}")
    rows = [("RMSE", rmse), ("MAE", mae), ("R²", r2)]
    for name, value in rows:
        print(f"      {name:<5} {YELLOW}{value:>10.4f}{RESET}")


def preview(df: pd.DataFrame, label: str, n: int = 8) -> None:
    """Print a compact preview of a dataframe (shape + first rows)."""
    rows, cols = df.shape
    print(f"  {BOLD}{label}{RESET} {DIM}({rows:,} rows x {cols} cols){RESET}")
    with pd.option_context(
        "display.max_columns", 12,
        "display.width", _WIDTH + 40,
        "display.max_rows", n,
    ):
        text = df.head(n).to_string()
    for line in text.splitlines():
        print(f"    {DIM}{line}{RESET}")
