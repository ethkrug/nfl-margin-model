"""End-to-end orchestration of the NFL point-margin model.

Run with ``python -m nfl_margin_model`` or ``python run_pipeline.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import advanced, console, data, features, model, qb, schedule


@dataclass
class Result:
    """Container for the trained model and the key intermediate frames."""

    model: object
    metrics: dict
    final: pd.DataFrame
    team_games_roll: pd.DataFrame
    replacement_epa: float


def run(show_previews: bool = True) -> Result:
    """Execute the full pipeline and return the trained model + artifacts."""
    console.rule("NFL Point-Margin Model")

    # --- 1. Load -------------------------------------------------------
    console.rule("1 · Loading data")
    _weekly_df, play_by_play_df = data.load_data()
    cols_with_epa, cols_with_wpa = data.epa_wpa_columns(play_by_play_df)
    console.info(
        f"discovered {len(cols_with_epa)} EPA cols, {len(cols_with_wpa)} WPA cols"
    )

    # --- 2. Game-level features ---------------------------------------
    console.rule("2 · Game & team aggregation")
    console.step("Collapsing play-by-play to game totals")
    game_totals = features.build_game_totals(play_by_play_df, cols_with_epa, cols_with_wpa)
    console.step("Summing per-team counting stats")
    team_game_sums = features.build_team_game_sums(play_by_play_df)
    console.step("Building game frame (target + weather imputation)")
    game_df = features.build_game_df(game_totals)
    if show_previews:
        console.preview(game_df, "game_df")

    # --- 3. Team-game reshape -----------------------------------------
    console.rule("3 · Team-game features")
    console.step("Reshaping into for_/allow_ team-game rows")
    team_games = features.build_team_games(game_df, team_game_sums)
    console.step("Computing advanced pbp metrics (red zone, explosive, pressure, ...)")
    adv = advanced.build_advanced_metrics(play_by_play_df)
    team_games = team_games.merge(adv, on=["game_id", "team"], how="left")
    # Neutral per-game fill for situations that didn't occur (e.g. no FG attempt,
    # no red-zone trip); these are pre-roll raw stats and get shifted next.
    adv_cols = [c for c in adv.columns if c not in ("game_id", "team")]
    team_games[adv_cols] = team_games[adv_cols].fillna(team_games[adv_cols].mean())
    console.step("Opponent-adjusting EPA (offense + defense) and rolling")
    team_games_roll = features.build_rolling_features(team_games)
    console.step("Attaching game context + weather interactions")
    precip = advanced.build_precip(play_by_play_df)
    team_games_roll = features.add_game_context(team_games_roll, game_df, precip)

    # --- 4. QB features ------------------------------------------------
    console.rule("4 · Starting-QB features")
    # Loaded before the depth charts: the new daily-snapshot schema needs the
    # kickoff dates to be mapped onto weeks. Reused by stage 4b below.
    schedules = data.load_schedules()
    depth_charts = data.load_depth_charts(schedules)
    injuries = data.load_injuries()
    team_games_roll, replacement_epa = qb.add_qb_features(
        team_games_roll, play_by_play_df, depth_charts, injuries
    )

    # --- 4b. Schedule/travel + matchup features -----------------------
    console.rule("4b · Schedule, travel & matchup features")
    console.step("Attaching rest / bye / short-week / travel / starters-out")
    team_games_roll = schedule.add_schedule_features(
        team_games_roll, schedules, depth_charts, injuries
    )
    console.step("Attaching opponent strength + matchup edges")
    team_games_roll = features.add_matchup_features(team_games_roll)

    # --- 5. Targets & model frame -------------------------------------
    console.rule("5 · Targets & model frame")
    console.step("Deriving score margin / win targets")
    team_games_roll, team_games = model.add_targets(team_games_roll, team_games, game_df)
    final = model.build_model_frame(team_games_roll)
    if show_previews:
        console.preview(final, "final model frame")

    # --- 6. Train & evaluate ------------------------------------------
    console.rule("6 · Training & evaluation")
    preprocessor = model.build_preprocessor()
    splits = model.split_data(final)
    console.info(
        f"train={len(splits['train'][0])}  "
        f"val={len(splits['val'][0])}  "
        f"test={len(splits['test'][0])} rows"
    )
    trained_model, metrics = model.train_and_evaluate(splits, preprocessor)

    console.rule("Done")
    return Result(
        model=trained_model,
        metrics=metrics,
        final=final,
        team_games_roll=team_games_roll,
        replacement_epa=replacement_epa,
    )


def inspect_teams(team_games_roll, teams=("NE", "SEA"), season=2025):
    """Return the rolled team-game rows for given teams/season (notebook cell 24)."""
    return team_games_roll[
        (team_games_roll["team"].isin(list(teams)))
        & (team_games_roll["season"] == season)
    ]


if __name__ == "__main__":
    run()
