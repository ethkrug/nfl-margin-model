"""Projected (not-yet-played) games for the upcoming season.

The core pipeline only builds rows for games that have play-by-play. To predict
upcoming games this module appends *shell* team-game rows for the next unplayed
week; the leakage-safe shift(1) rolling then seeds their features off completed
prior games. It also assigns each projected team its starting QB (from the
upcoming season's depth chart, falling back to the primary returning starter).
"""
from __future__ import annotations

import pandas as pd

from . import config, weather
from .qb import _normalize_depth


def projected_frames(sched, project_season, gdf, team_games, clim, ref_date):
    """Build game-context + shell team-game rows for the next unplayed week.

    Returns ``(context_rows, shell_rows, target_week)`` or ``(None, None, None)``.
    Only the *earliest fully-unplayed* week is projected, so its rolling features
    seed cleanly off completed prior games rather than off other projected weeks.
    """
    ps = sched[sched["season"] == project_season]
    if ps.empty:
        return None, None, None
    unplayed = [int(w) for w in sorted(ps["week"].unique())
                if not ps.loc[ps["week"] == w, "home_score"].notna().any()]
    if not unplayed:
        return None, None, None
    target = unplayed[0]
    wk = ps[ps["week"] == target].copy()

    ctx = pd.DataFrame({
        "game_id": wk["game_id"].values, "season": project_season, "week": target,
        "season_type": "REG", "div_game": wk["div_game"].fillna(0).astype(int).values,
        "roof": wk["roof"].values, "surface": wk["surface"].values,
        "gameday": wk["gameday"].values, "temp": pd.NA, "wind": pd.NA,
        "home_team": wk["home_team"].values, "away_team": wk["away_team"].values,
        "home_score": pd.NA, "away_score": pd.NA,
    })
    ctx = weather.impute_proj_weather(ctx, gdf, clim, ref_date)

    rows = []
    for _, r in wk.iterrows():
        rows.append((r["game_id"], r["home_team"], r["away_team"]))
        rows.append((r["game_id"], r["away_team"], r["home_team"]))
    key = pd.DataFrame(rows, columns=["game_id", "team", "opponent"])
    shell = key.reindex(columns=team_games.columns)
    shell["game_id"] = key["game_id"].values
    shell["team"] = key["team"].values
    shell["opponent"] = key["opponent"].values
    shell["season"] = project_season
    shell["week"] = target
    return ctx, shell, target


def depth_qb1(depth, project_season):
    """Projected-season QB1 per team from nflverse depth charts (latest week).

    Empty until nflverse publishes real offseason depth for the upcoming season
    (pre-season it mirrors last year), at which point trades/signings/rookies are
    picked up automatically with no code change.
    """
    nd = _normalize_depth(depth)
    nd = nd[(nd["season"] == project_season) & (nd["position"] == "QB")
            & (nd["depth_rank"] == 1)].dropna(subset=["week", "team", "player_id"])
    if nd.empty:
        return {}
    latest = nd.sort_values("week").groupby("team").tail(1)
    return dict(zip(latest["team"], latest["player_id"]))


def override_projected_qb(tgr, project_season, train_through, replacement, depth=None):
    """Assign each projected-season team its starting QB and that QB's quality.

    Starter, in priority order:
      1. the upcoming season's depth-chart QB1 (nflverse), once published;
      2. else the team's PRIMARY (most-starts) starter from the last completed
         season -- deliberately not most-recent, so an injured-late franchise QB
         (e.g. Mahomes) returns instead of the backup who finished the year.
    Quality = that QB's latest completed-season form; a QB with no NFL history
    (rookie) falls back to replacement level.
    """
    played = tgr[(tgr["season"] <= train_through) & tgr["qb_player_id"].notna()]
    if played.empty:
        return tgr
    hist = played.sort_values(["season", "week"]).groupby("qb_player_id").tail(1).set_index("qb_player_id")
    latest = tgr[(tgr["season"] == train_through) & tgr["qb_player_id"].notna()]
    latest = latest if not latest.empty else played
    counts = latest.groupby(["team", "qb_player_id"]).size().reset_index(name="n")
    primary = counts.sort_values("n").groupby("team").tail(1).set_index("team")["qb_player_id"].to_dict()
    dq1 = depth_qb1(depth, project_season) if depth is not None else {}

    for i in tgr.index[tgr["season"] == project_season]:
        t = tgr.at[i, "team"]
        pid = dq1.get(t) or primary.get(t)
        if pid is None:
            continue
        if pid in hist.index:
            L = hist.loc[pid]
            # Base = before any offseason pull, so applying r0 below cannot compound.
            q5 = L.get("qb_quality_base_5", L["qb_quality_5"])
            q10 = L.get("qb_quality_base_10", L["qb_quality_10"])
            ps = float(L.get("qb_prior_starts", 0) or 0) + 1
            # Season opener (0 current-season starts) -> full offseason regression
            # toward the QB's career baseline, matching the in-model treatment.
            r0 = config.QB_OFFSEASON_REG_STRENGTH
            career = L.get("qb_career_epa")
            if r0 and career is not None and not pd.isna(career):
                q5 = (1 - r0) * q5 + r0 * career
                q10 = (1 - r0) * q10 + r0 * career
        else:                              # rookie / no NFL history
            q5 = q10 = replacement
            ps = 0.0
        tgr.at[i, "qb_quality_5"] = q5
        tgr.at[i, "qb_quality_10"] = q10
        tgr.at[i, "qb_prior_starts"] = ps
        tgr.at[i, "qb_depth_order"] = 1
        tgr.at[i, "is_designated_starter"] = 1
        tgr.at[i, "starter_inactive"] = 0
        if "qb_player_id" in tgr.columns:
            tgr.at[i, "qb_player_id"] = pid
        if "qb_career_epa" in tgr.columns and pid in hist.index:
            tgr.at[i, "qb_career_epa"] = hist.loc[pid, "qb_career_epa"]
    return tgr
