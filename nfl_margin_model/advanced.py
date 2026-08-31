"""Advanced play-by-play team-game metrics.

These are per (game_id, team) rate/efficiency stats that the reshape + EPA
pipeline doesn't produce. They are merged onto ``team_games`` *before* the
rolling step so ``build_rolling_features`` rolls them leakage-safely (shift(1))
and drops the raw current-game value, exactly like the EPA/counting stats.

Offense-side (team with the ball, ``off_*``) and defense-side (team defending,
``def_*``) versions are produced for each metric so the matchup step can pair a
team's offense against the opponent's defense:

* ``off_expl_rate`` / ``def_expl_rate``     -- explosive-play rate (gained/allowed)
* ``off_rz_td_rate`` / ``def_rz_td_rate``   -- red-zone TD rate (scored/allowed)
* ``off_start_yl100`` / ``def_start_yl100`` -- avg drive-start field position
* ``off_pressure_rate`` / ``def_pressure_rate`` -- QB pressure (allowed/generated)
* ``off_fg_pct``                            -- field-goal make %
* ``off_fumbles`` / ``off_fumbles_lost``    -- ball-security
* ``def_fumbles_forced``                    -- fumbles the defense forced
* ``fum_recovery_rate``                     -- share of loose balls the team recovered

Also exposes ``build_precip`` for a game-level precipitation flag parsed from the
pbp ``weather`` text.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


# --------------------------------------------------------------------------
# Drive-level helpers (red-zone TD rate, starting field position)
# --------------------------------------------------------------------------
def _drive_metrics(pbp, team_col):
    """Per (game_id, ``team_col``) red-zone TD rate and avg drive-start yl100.

    ``team_col`` is ``posteam`` (offense) or ``defteam`` (defense). For defense
    the drives belong to the opponent, so this yields "allowed" versions.
    """
    d = pbp.dropna(subset=[team_col, "fixed_drive"]).copy()
    grp = d.groupby(["game_id", team_col, "fixed_drive"])

    drive = grp.agg(
        min_yardline=("yardline_100", "min"),
        result=("fixed_drive_result", "first"),
        # yardline_100 on the first snap of the drive = starting field position.
        start_yl100=("yardline_100", "first"),
    ).reset_index()

    drive["reached_rz"] = (drive["min_yardline"] <= config.REDZONE_YARDLINE).astype(int)
    drive["td"] = (drive["result"] == "Touchdown").astype(int)
    drive["rz_td"] = (drive["reached_rz"] & drive["td"]).astype(int)

    out = drive.groupby(["game_id", team_col]).agg(
        rz_drives=("reached_rz", "sum"),
        rz_tds=("rz_td", "sum"),
        start_yl100=("start_yl100", "mean"),
    ).reset_index()
    out["rz_td_rate"] = np.where(out["rz_drives"] > 0, out["rz_tds"] / out["rz_drives"], np.nan)
    return out.rename(columns={team_col: "team"})[
        ["game_id", "team", "rz_td_rate", "start_yl100"]
    ]


# --------------------------------------------------------------------------
# Play-level rates (explosive, pressure, FG, fumbles)
# --------------------------------------------------------------------------
def _offense_play_metrics(pbp):
    """Offense-side play rates keyed (game_id, posteam)."""
    p = pbp.dropna(subset=["posteam"]).copy()
    is_play = p["play_type"].isin(["run", "pass"]) & p["yards_gained"].notna()
    explosive = (
        (p["play_type"].eq("pass") & p["yards_gained"].ge(config.EXPLOSIVE_PASS_YARDS))
        | (p["play_type"].eq("run") & p["yards_gained"].ge(config.EXPLOSIVE_RUSH_YARDS))
    )
    p["_is_play"] = is_play.astype(float)
    p["_explosive"] = (is_play & explosive).astype(float)

    dbk = p["qb_dropback"].fillna(0).eq(1)
    p["_dropback"] = dbk.astype(float)
    p["_pressure"] = np.where(dbk, p["was_pressure"].fillna(0).astype(float), np.nan)

    fga = p["field_goal_attempt"].fillna(0).eq(1)
    p["_fga"] = fga.astype(float)
    p["_fg_made"] = np.where(fga, p["field_goal_result"].eq("made").astype(float), np.nan)

    p["_fumble"] = p["fumble"].fillna(0)
    p["_fumble_lost"] = p["fumble_lost"].fillna(0)

    g = p.groupby(["game_id", "posteam"])
    out = pd.DataFrame({
        "off_expl_rate": g["_explosive"].sum() / g["_is_play"].sum(),
        "off_pressure_rate": g["_pressure"].mean(),
        "off_fg_pct": g["_fg_made"].mean(),
        "off_fumbles": g["_fumble"].sum(),
        "off_fumbles_lost": g["_fumble_lost"].sum(),
    }).reset_index().rename(columns={"posteam": "team"})
    return out


def _defense_play_metrics(pbp):
    """Defense-side play rates keyed (game_id, defteam)."""
    p = pbp.dropna(subset=["defteam"]).copy()
    is_play = p["play_type"].isin(["run", "pass"]) & p["yards_gained"].notna()
    explosive = (
        (p["play_type"].eq("pass") & p["yards_gained"].ge(config.EXPLOSIVE_PASS_YARDS))
        | (p["play_type"].eq("run") & p["yards_gained"].ge(config.EXPLOSIVE_RUSH_YARDS))
    )
    p["_is_play"] = is_play.astype(float)
    p["_explosive"] = (is_play & explosive).astype(float)

    dbk = p["qb_dropback"].fillna(0).eq(1)
    p["_pressure"] = np.where(dbk, p["was_pressure"].fillna(0).astype(float), np.nan)
    p["_ff"] = p["fumble_forced"].fillna(0)

    g = p.groupby(["game_id", "defteam"])
    out = pd.DataFrame({
        "def_expl_rate": g["_explosive"].sum() / g["_is_play"].sum(),
        "def_pressure_rate": g["_pressure"].mean(),
        "def_fumbles_forced": g["_ff"].sum(),
    }).reset_index().rename(columns={"defteam": "team"})
    return out


def _fumble_recovery_rate(pbp):
    """Per (game_id, team) share of loose balls in the game the team recovered.

    Opportunities = every fumble in the team's game (either side); credit = a
    fumble whose recovering team is this team. Captures recovery luck rather than
    ball security.
    """
    f = pbp[pbp["fumble"].fillna(0).eq(1)].copy()
    # Teams that appear in each game (as posteam or defteam).
    teams = pd.concat([
        pbp[["game_id", "posteam"]].rename(columns={"posteam": "team"}),
        pbp[["game_id", "defteam"]].rename(columns={"defteam": "team"}),
    ]).dropna().drop_duplicates()

    fumbles_per_game = f.groupby("game_id").size().rename("game_fumbles")
    rec = (
        f.dropna(subset=["fumble_recovery_1_team"])
        .groupby(["game_id", "fumble_recovery_1_team"]).size()
        .rename("recovered").reset_index()
        .rename(columns={"fumble_recovery_1_team": "team"})
    )

    out = teams.merge(fumbles_per_game, on="game_id", how="left")
    out = out.merge(rec, on=["game_id", "team"], how="left")
    out["recovered"] = out["recovered"].fillna(0)
    out["fum_recovery_rate"] = np.where(
        out["game_fumbles"] > 0, out["recovered"] / out["game_fumbles"], np.nan
    )
    return out[["game_id", "team", "fum_recovery_rate"]]


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def build_advanced_metrics(pbp):
    """One row per (game_id, team) with all advanced off_/def_ metrics."""
    off_drive = _drive_metrics(pbp, "posteam").rename(
        columns={"rz_td_rate": "off_rz_td_rate", "start_yl100": "off_start_yl100"}
    )
    def_drive = _drive_metrics(pbp, "defteam").rename(
        columns={"rz_td_rate": "def_rz_td_rate", "start_yl100": "def_start_yl100"}
    )
    off_play = _offense_play_metrics(pbp)
    def_play = _defense_play_metrics(pbp)
    fum_rec = _fumble_recovery_rate(pbp)

    out = off_drive
    for frame in (def_drive, off_play, def_play, fum_rec):
        out = out.merge(frame, on=["game_id", "team"], how="outer")
    return out


def build_precip(pbp):
    """Game-level precipitation flag parsed from the pbp ``weather`` text."""
    if "weather" not in pbp.columns:
        return pd.DataFrame(columns=["game_id", "precip"])
    w = (
        pbp[["game_id", "weather"]]
        .dropna(subset=["weather"])
        .drop_duplicates("game_id")
        .copy()
    )
    text = w["weather"].str.lower()
    indoors = text.str.contains("indoor", na=False)
    has_precip = text.apply(
        lambda t: any(k in t for k in config.PRECIP_KEYWORDS)
    )
    w["precip"] = (has_precip & ~indoors).astype(int)
    return w[["game_id", "precip"]]
