"""NFL point-margin model pipeline.

A refactor of ``NFL_data3.ipynb`` into a clean, importable package. The public
entry point is :func:`nfl_margin_model.pipeline.run`.
"""

from . import config

__all__ = ["config", "pipeline"]
