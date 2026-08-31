#!/usr/bin/env python3
"""Entry point for the NFL point-margin model.

This is the script form of ``NFL_data3.ipynb``. It loads the data, engineers
the team/QB features, trains the tuned XGBoost model, and prints validation and
test metrics.

Usage:
    python run_pipeline.py
"""

from nfl_margin_model.pipeline import run, inspect_teams
from nfl_margin_model import console


def main():
    result = run()

    # Optional inspection of specific teams (notebook cell 24).
    sup_teams = inspect_teams(result.team_games_roll, teams=("NE", "SEA"), season=2025)
    console.rule("Inspection · NE / SEA (2025)")
    console.preview(sup_teams, "sup_teams")


if __name__ == "__main__":
    main()
