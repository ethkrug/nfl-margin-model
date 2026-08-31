"""Game-level and team-level feature engineering.

This mirrors the notebook's pipeline exactly:

1. collapse the play-by-play into one row per game (``game_totals``) and one row
   per team-game of counting stats (``team_game_sums``),
2. build ``game_df`` (target + cleaned context, weather imputation),
3. reshape into one row per team-game with ``for_`` / ``allow_`` stats,
4. opponent-adjust offensive EPA and roll all stats over prior games, and
5. attach game context.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


# --------------------------------------------------------------------------
# 1. Collapse play-by-play
# --------------------------------------------------------------------------
def build_game_totals(play_by_play_df, cols_with_epa, cols_with_wpa):
    """One row per game: end-of-game cumulative ``total_*`` EPA/WPA + context."""
    epa_wpa = cols_with_epa + cols_with_wpa
    totals_list = (
        ["game_id", "play_id"]
        + [c for c in epa_wpa if "total" in c]
        + ["home_team", "away_team"]
        + config.GAME_FEATURES
    )
    game_totals = (
        play_by_play_df[totals_list]
        .sort_values(["game_id", "play_id"])
        .groupby("game_id", as_index=False)
        .last()  # last non-null per column
        .drop("play_id", axis=1)
    )
    return game_totals


def build_team_game_sums(play_by_play_df):
    """One row per (game, possession team): summed counting stats."""
    sum_cols = ["game_id", "posteam"] + list(dict.fromkeys(config.SUM_COLS))
    return (
        play_by_play_df[sum_cols]
        .groupby(["game_id", "posteam"], as_index=False)
        .sum(numeric_only=True)
    )


# --------------------------------------------------------------------------
# 2. game_df: target + cleaned context
# --------------------------------------------------------------------------
def build_game_df(game_totals):
    """Add the score-margin target, reorder columns, and impute weather."""
    game_df = game_totals.copy()
    game_df["y"] = game_df["home_score"] - game_df["away_score"]

    # Move the key identifying columns to the front, in a stable order.
    cols = list(game_df.columns)
    cols.insert(1, cols.pop(cols.index("season")))
    cols.insert(2, cols.pop(cols.index("week")))
    cols.insert(3, cols.pop(cols.index("season_type")))
    cols.insert(4, cols.pop(cols.index("home_team")))
    cols.insert(5, cols.pop(cols.index("away_team")))
    game_df = game_df[cols]

    # Indoor games get fixed, neutral weather.
    game_df.loc[game_df["roof"].isin(["dome", "closed"]), ["temp", "wind"]] = [70.0, 0.0]

    # Coerce to numeric, then impute remaining gaps with the home stadium's mean.
    game_df["temp"] = pd.to_numeric(game_df["temp"], errors="coerce")
    game_df["wind"] = pd.to_numeric(game_df["wind"], errors="coerce")
    game_df["temp"] = game_df["temp"].fillna(
        game_df.groupby("home_team")["temp"].transform("mean")
    )
    game_df["wind"] = game_df["wind"].fillna(
        game_df.groupby("home_team")["wind"].transform("mean")
    )
    return game_df


# --------------------------------------------------------------------------
# 3. Reshape to one row per team-game (for_ / allow_)
# --------------------------------------------------------------------------
def build_team_games(game_df, team_game_sums):
    """Stack home and away perspectives into one row per team-game.

    Each team-game gets ``for_*`` (offense) and ``allow_*`` (defense) totals,
    plus the team's summed counting stats.
    """
    key_cols = ["game_id", "season", "week"]
    total_cols = [
        c for c in game_df.columns
        if c.startswith("total_home_") or c.startswith("total_away_")
    ]
    home_totals = [c for c in total_cols if c.startswith("total_home_")]
    away_totals = [c for c in total_cols if c.startswith("total_away_")]

    # HOME ROWS: offense = home totals; allowed = away totals.
    home = game_df[key_cols + ["home_team", "away_team"] + home_totals + away_totals].copy()
    home = home.rename(columns={"home_team": "team", "away_team": "opponent"})
    home = home.rename(columns={c: "for_" + c.replace("total_home_", "") for c in home_totals})
    home = home.rename(columns={c: "allow_" + c.replace("total_away_", "") for c in away_totals})

    # AWAY ROWS: offense = away totals; allowed = home totals.
    away = game_df[key_cols + ["away_team", "home_team"] + away_totals + home_totals].copy()
    away = away.rename(columns={"away_team": "team", "home_team": "opponent"})

    away.columns = away.columns.str.replace("total_away_", "for_").str.replace("total_home_", "allow_")
    home.columns = home.columns.str.replace("total_home_", "for_").str.replace("total_away_", "allow_")

    team_games = pd.concat([home, away], ignore_index=True).sort_values(["team", "season", "week"])

    team_games = team_games.merge(
        team_game_sums,
        left_on=["game_id", "team"],
        right_on=["game_id", "posteam"],
        how="left",
    ).drop(columns=["posteam"])
    return team_games


# --------------------------------------------------------------------------
# 4. Opponent-adjusted EPA + rolling features
# --------------------------------------------------------------------------
def build_rolling_features(team_games, windows=config.ROLL_WINDOWS, adjust_defense=True,
                           prior_weight=config.roll_prior_weight, seed_shrink=1.0):
    """Opponent-adjust EPA (offense AND defense), then roll all stats.

    Leakage-safe: every rolling/expanding mean is ``shift(1)``-ed so a row only
    sees games that happened before it. The previous season's mean seeds each
    team-season so week-1 features are not empty.

    Adjustment is symmetric:

    * offensive EPA is credited relative to the opponent defense's prior EPA
      *allowed* (``for_x - opp_avg_allow_x``), and
    * defensive EPA is credited relative to the opponent offense's prior EPA
      *produced* (``allow_x - opp_avg_for_x``), so a defense isn't flattered by
      having faced weak offenses.

    ``adjust_defense=False`` rolls the raw ``allow_*`` EPA instead (the original
    behaviour), used to isolate the defensive-adjustment change in evaluation.
    """
    tg = team_games.sort_values(["team", "season", "week"]).reset_index(drop=True)
    tg = tg.loc[:, ~tg.columns.duplicated()].copy()

    id_cols = ["game_id", "season", "week", "team", "opponent"]
    base_stat_cols = [c for c in tg.columns if c not in id_cols]

    # Offensive EPA columns and their matching "allowed" counterparts.
    epa_for_cols = [c for c in base_stat_cols if c.startswith("for_") and c.endswith("_epa")]
    allow_epa_cols = sorted(
        {
            f"allow_{c[len('for_'):]}"
            for c in epa_for_cols
            if f"allow_{c[len('for_'):]}" in tg.columns
        }
    )
    # Opponent-strength columns: prior "allowed" (defense faced by our offense)
    # and prior "for" (offense faced by our defense).
    opp_allow_cols = [f"opp_avg_{c}" for c in allow_epa_cols]
    opp_for_cols = [f"opp_avg_{c}" for c in epa_for_cols] if adjust_defense else []
    opponent_strength_cols = opp_allow_cols + opp_for_cols

    def _prior_frame(src_cols, out_cols):
        """Per (team,season) expanding-mean-then-shift prior of ``src_cols``.

        Seeded by the team's prior-season mean so week 1 is populated. Returns a
        frame keyed (game_id, team) with columns renamed to ``out_cols``.
        """
        prev_means = (
            tg.groupby(["team", "season"])[src_cols].mean().groupby(level=0).shift(1)
        )

        def compute(g, team, season):
            g = g.sort_values("week").copy()
            prev = prev_means.loc[(team, season)].fillna(0).astype(float)
            vals = g[src_cols].astype(float).reset_index(drop=True)
            ext = pd.concat([prev.to_frame().T, vals], ignore_index=True)
            prior = (
                ext.expanding(min_periods=1).mean().shift(1).iloc[1:].reset_index(drop=True)
            )
            prior.index = g.index
            prior.columns = out_cols
            return pd.concat([g[["game_id", "team"]], prior], axis=1)

        return pd.concat(
            [compute(g, team, season)
             for (team, season), g in tg.groupby(["team", "season"], sort=False)],
            ignore_index=True,
        )

    # Prior defensive strength (EPA allowed) and, when adjusting defense, prior
    # offensive strength (EPA for).
    team_strength_prior = _prior_frame(allow_epa_cols, opp_allow_cols)
    if adjust_defense:
        team_for_prior = _prior_frame(epa_for_cols, opp_for_cols)
        team_strength_prior = team_strength_prior.merge(
            team_for_prior, on=["game_id", "team"], how="outer"
        )

    # Attach each opponent's prior strength to this team-game (join on opponent).
    tg = tg.merge(
        team_strength_prior,
        left_on=["game_id", "opponent"],
        right_on=["game_id", "team"],
        how="left",
        suffixes=("", "_opponent_strength"),
    ).drop(columns=["team_opponent_strength"])
    tg[opponent_strength_cols] = tg[opponent_strength_cols].fillna(0)

    adjusted_epa_cols = []
    # Opponent-adjusted offensive EPA = achieved - opponent's prior EPA allowed.
    for for_col in epa_for_cols:
        avg_col = f"opp_avg_allow_{for_col[len('for_'):]}"
        if avg_col in tg.columns:
            adjusted_col = f"{for_col}_opp_adjusted"
            tg[adjusted_col] = tg[for_col].astype(float) - tg[avg_col].astype(float)
            adjusted_epa_cols.append(adjusted_col)
    # Opponent-adjusted defensive EPA = allowed - opponent's prior EPA produced.
    if adjust_defense:
        for allow_col in allow_epa_cols:
            avg_col = f"opp_avg_for_{allow_col[len('allow_'):]}"
            if avg_col in tg.columns:
                adjusted_col = f"{allow_col}_opp_adjusted"
                tg[adjusted_col] = tg[allow_col].astype(float) - tg[avg_col].astype(float)
                adjusted_epa_cols.append(adjusted_col)

    # Roll the adjusted offensive EPA (and, if enabled, adjusted defensive EPA)
    # instead of the raw ones. When defense isn't adjusted, raw allow_ EPA rolls.
    rolled_out_epa = epa_for_cols + (allow_epa_cols if adjust_defense else [])
    stat_cols = [
        c for c in tg.columns
        if c not in id_cols + rolled_out_epa + opponent_strength_cols
    ]

    prev_season_means = (
        tg.groupby(["team", "season"])[stat_cols]
        .mean()
        .groupby(level=0)
        .shift(1)
    )

    # Optionally regress the prior-season seed toward the league mean of that
    # prior season: seed = league + shrink*(team - league). shrink=1 keeps the
    # raw prior rating (no-op); <1 pulls extreme ratings back toward average;
    # ``seed_shrink`` may be a scalar or a callable ``col -> shrink`` so luck
    # stats can be regressed harder than skill stats.
    if callable(seed_shrink):
        shrink = np.array([float(seed_shrink(c)) for c in stat_cols])
    else:
        shrink = np.full(len(stat_cols), float(seed_shrink))
    if not np.allclose(shrink, 1.0):
        league_mean = prev_season_means.groupby(level=1).transform("mean")
        seed_means = league_mean + (prev_season_means - league_mean) * shrink
    else:
        seed_means = prev_season_means

    def compute_rolls(g, team, season, windows=windows):
        """Prior-seeded rolling means with a tunable prior weight (no leakage).

        For each game the feature blends the previous-season seed with the
        current-season games already played, capped at a ``w``-game window::

            value = (prior_weight*empty*seed + sum(current in window))
                    / (prior_weight*empty + count)

        where ``count`` is current-season games in the window and
        ``empty = w - count`` are the slots the prior seed fills. Once the window
        fills with current games (``empty == 0``) the prior drops out and the
        value is the plain current-game mean -- so only the first ``w`` weeks
        change. ``prior_weight`` may be a scalar or a callable ``week -> weight``
        so the prior can be stronger in the noisy first few weeks than later.
        """
        g = g.sort_values("week").copy()
        g["team"] = team
        g["season"] = season

        seed = seed_means.loc[(team, season)].fillna(0).values.astype(float)
        stats = g[stat_cols].to_numpy(dtype=float)
        weeks = g["week"].to_numpy()
        pw_call = callable(prior_weight)
        n, k_cols = stats.shape
        # csum[i] = sum of current-season games g_0..g_{i-1} (NaN treated as 0).
        csum = np.vstack([np.zeros(k_cols), np.nancumsum(stats, axis=0)])

        roll_frames = []
        for w in windows:
            out = np.empty((n, k_cols))
            for idx in range(n):
                wp = prior_weight(weeks[idx]) if pw_call else prior_weight
                count = min(idx, w)                 # current games before this one
                empty = w - count                   # window slots filled by prior
                cur_sum = csum[idx] - csum[idx - count]
                denom = wp * empty + count
                out[idx] = seed if denom <= 0 else (wp * empty * seed + cur_sum) / denom
            roll_frames.append(
                pd.DataFrame(
                    out,
                    columns=[f"{c}_roll{w}" for c in stat_cols],
                    index=g.index,  # keep alignment with g
                )
            )

        # One concat instead of repeated wide inserts -> no fragmentation.
        return pd.concat([g, *roll_frames], axis=1)

    team_games_roll = pd.concat(
        [
            compute_rolls(g, team, season, windows=windows)
            for (team, season), g in tg.groupby(["team", "season"], sort=False)
        ],
        ignore_index=True,
    )

    drop_after_roll = stat_cols + rolled_out_epa + opponent_strength_cols
    team_games_roll = team_games_roll.drop(columns=drop_after_roll, errors="ignore")
    return team_games_roll


# --------------------------------------------------------------------------
# 5. Attach game context
# --------------------------------------------------------------------------
def add_game_context(team_games_roll, game_df, precip=None):
    """Merge weather/venue context, a home/away flag, and weather interactions.

    Keeps the existing raw ``temp``/``wind`` columns and adds:

    * ``precip`` -- game-level precipitation flag (0 when unavailable/indoor),
    * ``home_x_temp`` / ``home_x_wind`` / ``home_x_precip`` -- home teams are
      acclimated to their own stadium's conditions, so weather should hurt the
      visitor more,
    * ``temp_x_wind`` / ``wind_x_precip`` -- compounding bad-weather effects that
      most punish passing and kicking.
    """
    game_ctx_cols = [
        "game_id", "div_game", "season_type", "roof", "surface",
        "temp", "wind", "home_team",
    ]
    team_games_roll = team_games_roll.merge(game_df[game_ctx_cols], on="game_id", how="left")

    team_games_roll["is_home"] = (
        team_games_roll["team"] == team_games_roll["home_team"]
    ).astype(int)
    team_games_roll = team_games_roll.drop(columns=["home_team"])

    # Precipitation flag (0 where no weather text / indoor).
    if precip is not None and len(precip):
        team_games_roll = team_games_roll.merge(precip, on="game_id", how="left")
    if "precip" not in team_games_roll.columns:
        team_games_roll["precip"] = 0
    team_games_roll["precip"] = team_games_roll["precip"].fillna(0).astype(int)

    # Weather interactions (raw temp/wind are retained above).
    temp = team_games_roll["temp"].astype(float)
    wind = team_games_roll["wind"].astype(float)
    home = team_games_roll["is_home"].astype(float)
    prcp = team_games_roll["precip"].astype(float)
    team_games_roll["home_x_temp"] = home * temp
    team_games_roll["home_x_wind"] = home * wind
    team_games_roll["home_x_precip"] = home * prcp
    team_games_roll["temp_x_wind"] = temp * wind
    team_games_roll["wind_x_precip"] = wind * prcp

    # Re-order: identifiers, then context, then everything else.
    id_cols = ["game_id", "season", "week", "team", "is_home"]
    ctx_cols = [
        "div_game", "season_type", "roof", "surface", "temp", "wind", "precip",
        "home_x_temp", "home_x_wind", "home_x_precip", "temp_x_wind", "wind_x_precip",
    ]
    other_cols = [c for c in team_games_roll.columns if c not in id_cols + ctx_cols]
    return team_games_roll[id_cols + ctx_cols + other_cols]


# --------------------------------------------------------------------------
# 6. Matchup / opponent features
# --------------------------------------------------------------------------
def add_matchup_features(team_games_roll, windows=config.ROLL_WINDOWS):
    """Attach the *opponent's* rolled strength and build matchup edges.

    The base pipeline gives each row only its own team's rolled form. To predict
    a margin the model needs both sides, so this self-joins the frame on
    ``opponent`` to bring over every ``opp_*`` rolled stat and QB-quality column,
    then engineers explicitly-signed matchup edges (higher = better for *this*
    team) that a shallow tree would otherwise have to discover on its own:

    * ``edge_off_<epa>``  -- my (opp-adjusted) offense + opponent's leaky defense
    * ``edge_def_<epa>``  -- my leaky defense + opponent's offense (higher = worse)
    * ``edge_net_<epa>``  -- offensive edge minus defensive edge
    * ``edge_qb_quality_<w>`` -- my starting-QB quality minus the opponent's
    """
    tgr = team_games_roll
    roll_suffixes = tuple(f"_roll{w}" for w in windows)

    qb_cols = [
        c for c in (
            [f"qb_quality_{w}" for w in config.QB_WINDOWS]
            + ["qb_depth_order", "is_designated_starter", "starter_inactive",
               "qb_prior_starts"]
        )
        if c in tgr.columns
    ]
    opp_source = list(dict.fromkeys(
        [c for c in tgr.columns if c.endswith(roll_suffixes)] + qb_cols
    ))

    opp_view = (
        tgr[["game_id", "team"] + opp_source]
        .rename(columns={"team": "opponent"})
        .rename(columns={c: f"opp_{c}" for c in opp_source})
    )
    tgr = tgr.merge(opp_view, on=["game_id", "opponent"], how="left")

    new = {}
    # Opponent-adjusted EPA matchup edges (correct signs; see docstring).
    for w in windows:
        suf = f"_opp_adjusted_roll{w}"
        for fc in [c for c in tgr.columns if c.startswith("for_") and c.endswith(suf)]:
            base = fc[len("for_"):-len(suf)]
            ac, opp_fc, opp_ac = f"allow_{base}{suf}", f"opp_{fc}", f"opp_allow_{base}{suf}"
            if ac in tgr.columns and opp_ac in tgr.columns and opp_fc in tgr.columns:
                off_edge = tgr[fc].astype(float) + tgr[opp_ac].astype(float)
                def_edge = tgr[ac].astype(float) + tgr[opp_fc].astype(float)
                new[f"edge_off_{base}_roll{w}"] = off_edge
                new[f"edge_def_{base}_roll{w}"] = def_edge
                new[f"edge_net_{base}_roll{w}"] = off_edge - def_edge

    # Starting-QB quality edge (both higher = better, so a plain difference).
    for w in config.QB_WINDOWS:
        col = f"qb_quality_{w}"
        if col in tgr.columns and f"opp_{col}" in tgr.columns:
            new[f"edge_qb_quality_{w}"] = tgr[col].astype(float) - tgr[f"opp_{col}"].astype(float)

    if new:
        tgr = pd.concat([tgr, pd.DataFrame(new, index=tgr.index)], axis=1)

    # The opponent's raw rolled stats are redundant with the edge_* differentials
    # (which are built from them) and only add overfitting noise, so drop them --
    # keeping the opponent QB columns. Backtests: dropping these improves RMSE.
    if not config.KEEP_OPP_RAW:
        drop_opp = [c for c in tgr.columns
                    if c.startswith("opp_") and not c.startswith("opp_qb")]
        tgr = tgr.drop(columns=drop_opp)

    # Own-team rolled EPA/WPA are redundant with the edge_* EPA differentials
    # (built from them just above); drop them too. Backtested on two holdouts.
    if not config.KEEP_OWN_EPA_WPA:
        drop_own = [c for c in tgr.columns
                    if c.startswith(("for_", "allow_")) and "_roll" in c
                    and ("epa" in c or "wpa" in c)]
        tgr = tgr.drop(columns=drop_own)
    return tgr
