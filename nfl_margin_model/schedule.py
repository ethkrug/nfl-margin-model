"""Schedule, travel, and availability context features.

Everything here is knowable *before* kickoff, so no rolling/leakage handling is
needed -- these are merged straight onto the model frame:

* **Rest & week shape** -- ``rest_days`` (days since the team's last game),
  ``rest_diff`` (this team minus opponent), ``is_off_bye`` and ``is_short_week``
  flags. Rest is a well-known margin driver (off-a-bye edge, Thursday short
  weeks).
* **Travel** -- ``tz_shift`` (signed hours the team's body clock is displaced by
  the game's location) and ``abs_tz_shift``. West-coast teams in early
  East-coast kickoffs are the classic effect.
* **Neutral site** -- ``neutral_site`` flag (London/Germany/Mexico/Super Bowls),
  which negates the home-field edge ``is_home`` would otherwise imply.
* **Availability** -- ``starters_out``: count of depth-chart starters (offense +
  defense) listed Out/Doubtful on the injury report, i.e. non-QB injury load.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, console
from .data import _pos_family
from .qb import _normalize_depth

# Pro Bowl weekly injury-load columns (see build_pb_injury_features).
PB_INJ_COLS = [
    "pb_out_ever", "pb_out_recent", "pb_out_elite",
    "pb_out_ever_nonqb", "pb_out_recent_nonqb", "pb_out_elite_nonqb",
    "pb_out_ol",
]


# --------------------------------------------------------------------------
# Time-zone helpers
# --------------------------------------------------------------------------
def _game_tz_offset(row):
    """TZ offset (hours behind ET) of where the game is actually played."""
    if str(row.get("location", "Home")) == "Neutral":
        stadium = str(row.get("stadium", "")).lower()
        for key, off in config.INTL_STADIUM_TZ.items():
            if key in stadium:
                return off
        # Domestic neutral site (Super Bowl, relocation): fall back to the
        # nominal home team's home tz.
    return config.TEAM_TZ.get(row.get("home_team"), 0)


# --------------------------------------------------------------------------
# Rest / travel / neutral, one row per (game_id, team)
# --------------------------------------------------------------------------
def build_schedule_features(schedules):
    """Per team-game rest, travel and neutral-site context.

    Returns a frame keyed on ``["game_id", "team"]``.
    """
    s = schedules.copy()
    s["location"] = s["location"].fillna("Home")
    s["game_tz"] = s.apply(_game_tz_offset, axis=1)
    s["neutral_site"] = (s["location"] == "Neutral").astype(int)

    rows = []
    for side, opp_side in (("home", "away"), ("away", "home")):
        team_col = f"{side}_team"
        rest_col = f"{side}_rest"
        opp_rest_col = f"{opp_side}_rest"
        d = pd.DataFrame({
            "game_id": s["game_id"],
            "team": s[team_col],
            "rest_days": pd.to_numeric(s[rest_col], errors="coerce"),
            "rest_diff": pd.to_numeric(s[rest_col], errors="coerce")
            - pd.to_numeric(s[opp_rest_col], errors="coerce"),
            "neutral_site": s["neutral_site"],
            # Body-clock shift: game location tz minus the team's home tz.
            # Home team in a normal game -> 0. Positive = travelled west.
            "tz_shift": s["game_tz"] - s[team_col].map(config.TEAM_TZ).fillna(0),
        })
        rows.append(d)

    out = pd.concat(rows, ignore_index=True)
    out["abs_tz_shift"] = out["tz_shift"].abs()
    out["is_short_week"] = (out["rest_days"] <= config.SHORT_WEEK_MAX_REST).astype(int)
    out["is_off_bye"] = (out["rest_days"] >= config.OFF_BYE_MIN_REST).astype(int)
    return out[
        ["game_id", "team", "rest_days", "rest_diff", "is_short_week",
         "is_off_bye", "neutral_site", "tz_shift", "abs_tz_shift"]
    ]


# --------------------------------------------------------------------------
# Starters-out (non-QB injury load), one row per (season, week, team)
# --------------------------------------------------------------------------
def build_starters_out(depth_charts, injuries):
    """Count of depth-chart starters (offense+defense) ruled Out/Doubtful.

    A "starter" is a player at depth rank 1 in the Offense or Defense unit that
    week. This is the injury-report load beyond the QB features and is a
    leakage-safe pre-game feed.
    """
    # _normalize_depth already exposes season/week/team/player_id/depth_rank
    # and the "Offense"/"Defense"/"Special Teams" formation.
    dc = _normalize_depth(depth_charts)
    starters = dc[
        (dc["depth_rank"] == 1)
        & (dc["formation"].isin(["Offense", "Defense"]))
    ].dropna(subset=["season", "week", "team", "player_id"]).copy()
    starters["season"] = starters["season"].astype(int)
    starters["week"] = starters["week"].astype(int)
    starters = starters.drop_duplicates(["season", "week", "team", "player_id"])

    inj = injuries[["season", "week", "team", "gsis_id", "report_status"]].copy()
    inj["season"] = pd.to_numeric(inj["season"], errors="coerce")
    inj["week"] = pd.to_numeric(inj["week"], errors="coerce")
    inj = inj.dropna(subset=["season", "week"])
    inj["season"] = inj["season"].astype(int)
    inj["week"] = inj["week"].astype(int)
    inj["is_out"] = inj["report_status"].isin(config.STARTER_OUT_STATUSES).astype(int)
    inj = inj.rename(columns={"gsis_id": "player_id"})
    inj = inj.drop_duplicates(["season", "week", "team", "player_id"])

    merged = starters.merge(
        inj[["season", "week", "team", "player_id", "is_out"]],
        on=["season", "week", "team", "player_id"],
        how="left",
    )
    merged["is_out"] = merged["is_out"].fillna(0)

    out = (
        merged.groupby(["season", "week", "team"], as_index=False)["is_out"]
        .sum()
        .rename(columns={"is_out": "starters_out"})
    )
    return out


# --------------------------------------------------------------------------
# Pro Bowl pedigree: weekly injury load + season-transition churn
# --------------------------------------------------------------------------
def _pb_eligibility(pro_bowlers, season):
    """Pro Bowl pedigree cohorts for ``season``, from PRIOR seasons only.

    Returns ``(ever, recent, elite)`` sets of gsis ids:
    * **ever**   -- selected at any point before this season
    * **recent** -- selected within the last ``PB_RECENT_YEARS`` seasons
    * **elite**  -- selected >= ``PB_ELITE_COUNT`` times in the last
      ``PB_ELITE_WINDOW`` seasons ("x times in y years")

    Using only seasons < ``season`` keeps every derived feature leakage-safe.
    """
    ever, recent, counts = set(), set(), {}
    for s, players in pro_bowlers.items():
        if s >= season:
            continue
        ever |= players
        if s >= season - config.PB_RECENT_YEARS:
            recent |= players
        if s >= season - config.PB_ELITE_WINDOW:
            for g in players:
                counts[g] = counts.get(g, 0) + 1
    elite = {g for g, c in counts.items() if c >= config.PB_ELITE_COUNT}
    return ever, recent, elite


def build_pb_injury_features(injuries, pro_bowlers):
    """Weekly count of ruled-out players carrying Pro Bowl pedigree.

    ``starters_out`` treats every unavailable starter alike, so losing a
    perennial Pro Bowl tackle scores the same as losing a backup-grade one.
    These columns weight that injury load by pedigree, keyed on
    ``["season", "week", "team"]``:

    * ``pb_out_ever``   -- ruled-out players who were ever a Pro Bowler
    * ``pb_out_recent`` -- ... within the last ``PB_RECENT_YEARS`` seasons
    * ``pb_out_elite``  -- ... ``PB_ELITE_COUNT``+ times in ``PB_ELITE_WINDOW`` years

    Each also gets a ``_nonqb`` variant: a missing Pro Bowl quarterback is
    already the model's strongest signal (the QB features re-rate the team to
    the backup), so counting them here double-counts. The ``_nonqb`` columns
    isolate the information the QB features *don't* already carry. ``pb_out_ol``
    narrows that further to the offensive line, the most plausible non-QB
    mechanism for a single absence to swing a margin.

    The injury report is a pre-game feed and pedigree uses prior seasons only,
    so this is leakage-safe.
    """
    inj = injuries[
        ["season", "week", "team", "gsis_id", "report_status", "position"]
    ].copy()
    inj["season"] = pd.to_numeric(inj["season"], errors="coerce")
    inj["week"] = pd.to_numeric(inj["week"], errors="coerce")
    inj = inj.dropna(subset=["season", "week", "team", "gsis_id"])
    inj["season"] = inj["season"].astype(int)
    inj["week"] = inj["week"].astype(int)
    inj = inj[inj["report_status"].isin(config.STARTER_OUT_STATUSES)]
    inj = inj.drop_duplicates(["season", "week", "team", "gsis_id"])

    cols = PB_INJ_COLS
    frames = []
    for season, g in inj.groupby("season", sort=False):
        ever, recent, elite = _pb_eligibility(pro_bowlers, season)
        g = g.copy()
        pos = g["position"].astype(str).str.upper()
        not_qb = (pos != "QB").astype(int)
        is_ol = pos.map(lambda p: _pos_family(p) == "OL").astype(int)
        for tag, cohort in (("ever", ever), ("recent", recent), ("elite", elite)):
            member = g["gsis_id"].isin(cohort).astype(int)
            g[f"pb_out_{tag}"] = member
            g[f"pb_out_{tag}_nonqb"] = member * not_qb
        g["pb_out_ol"] = g["gsis_id"].isin(ever).astype(int) * is_ol
        frames.append(
            g.groupby(["season", "week", "team"], as_index=False)[cols].sum()
        )
    if not frames:
        return pd.DataFrame(columns=["season", "week", "team"] + list(cols))
    return pd.concat(frames, ignore_index=True)


def build_pb_churn_features(depth_charts, pro_bowlers):
    """Offseason arrivals/departures of players carrying Pro Bowl pedigree.

    Team membership comes from each season's **week-1 depth chart**, so a
    "departure" means the player opened the prior season on this team's chart
    and does not open this one there. Keyed on ``["season", "team"]``:
    ``pb_lost_*`` / ``pb_gain_*`` / ``pb_net_*`` (net = gained - lost) for the
    *ever* and *recent* cohorts, plus net for the *elite* cohort.
    """
    dc = _normalize_depth(depth_charts)
    dc = dc.dropna(subset=["season", "week", "team", "player_id"]).copy()
    dc["season"] = pd.to_numeric(dc["season"], errors="coerce")
    dc["week"] = pd.to_numeric(dc["week"], errors="coerce")
    dc = dc.dropna(subset=["season", "week"])
    wk1 = dc[dc["week"].astype(int) == 1]

    membership = {
        (int(season), team): set(g["player_id"])
        for (season, team), g in wk1.groupby(["season", "team"], sort=False)
    }
    seasons = sorted({s for s, _ in membership})
    teams = sorted({t for _, t in membership})

    rows = []
    for season in seasons:
        ever, recent, elite = _pb_eligibility(pro_bowlers, season)
        for team in teams:
            prev = membership.get((season - 1, team))
            now = membership.get((season, team))
            if not prev or not now:
                continue
            left, joined = prev - now, now - prev
            rows.append((
                season, team,
                len(left & ever), len(joined & ever),
                len(left & recent), len(joined & recent),
                len(left & elite), len(joined & elite),
            ))
    out = pd.DataFrame(rows, columns=[
        "season", "team", "pb_lost_ever", "pb_gain_ever",
        "pb_lost_recent", "pb_gain_recent", "pb_lost_elite", "pb_gain_elite",
    ])
    for tag in ("ever", "recent", "elite"):
        out[f"pb_net_{tag}"] = out[f"pb_gain_{tag}"] - out[f"pb_lost_{tag}"]
    return out


def add_pb_features(team_games_roll, injuries, depth_charts, pro_bowlers):
    """Merge Pro Bowl injury-load + churn columns and build opponent edges.

    Edges are signed so higher = better for *this* team: the opponent missing
    Pro Bowlers helps me, my own net Pro Bowl additions help me.
    """
    inj_cols = list(PB_INJ_COLS)
    churn_cols = [
        "pb_lost_ever", "pb_gain_ever", "pb_net_ever",
        "pb_lost_recent", "pb_gain_recent", "pb_net_recent",
        "pb_lost_elite", "pb_gain_elite", "pb_net_elite",
    ]

    inj_feats = build_pb_injury_features(injuries, pro_bowlers)
    team_games_roll = team_games_roll.merge(
        inj_feats, on=["season", "week", "team"], how="left"
    )
    churn_feats = build_pb_churn_features(depth_charts, pro_bowlers)
    team_games_roll = team_games_roll.merge(
        churn_feats, on=["season", "team"], how="left"
    )
    for col in inj_cols + churn_cols:
        team_games_roll[col] = team_games_roll[col].fillna(0.0)

    if "opponent" not in team_games_roll.columns:
        return team_games_roll

    # Opponent's values, joined on the shared game, for the differentials.
    opp_inj = (
        team_games_roll[["game_id", "team"] + inj_cols + churn_cols]
        .rename(columns={"team": "opponent"})
        .rename(columns={c: f"opp_{c}" for c in inj_cols + churn_cols})
    )
    team_games_roll = team_games_roll.merge(
        opp_inj, on=["game_id", "opponent"], how="left"
    )
    for col in inj_cols + churn_cols:
        team_games_roll[f"opp_{col}"] = team_games_roll[f"opp_{col}"].fillna(0.0)

    edges = {}
    for col in inj_cols:                     # their injuries help me
        edges[f"edge_{col}"] = team_games_roll[f"opp_{col}"] - team_games_roll[col]
    for tag in ("ever", "recent", "elite"):  # my net additions help me
        c = f"pb_net_{tag}"
        edges[f"edge_{c}"] = team_games_roll[c] - team_games_roll[f"opp_{c}"]
    team_games_roll = pd.concat(
        [team_games_roll, pd.DataFrame(edges, index=team_games_roll.index)], axis=1
    )
    return team_games_roll.drop(
        columns=[f"opp_{c}" for c in inj_cols + churn_cols]
    )


def add_schedule_features(team_games_roll, schedules, depth_charts, injuries,
                          pro_bowlers=None):
    """Merge rest/travel/neutral + starters-out onto the model frame."""
    sched_feats = build_schedule_features(schedules)
    team_games_roll = team_games_roll.merge(
        sched_feats, on=["game_id", "team"], how="left"
    )

    starters_out = build_starters_out(depth_charts, injuries)
    team_games_roll = team_games_roll.merge(
        starters_out, on=["season", "week", "team"], how="left"
    )

    # Sensible pre-game defaults where a feed is missing.
    team_games_roll["rest_days"] = team_games_roll["rest_days"].fillna(7)
    team_games_roll["rest_diff"] = team_games_roll["rest_diff"].fillna(0)
    for col in ["is_short_week", "is_off_bye", "neutral_site"]:
        team_games_roll[col] = team_games_roll[col].fillna(0).astype(int)
    for col in ["tz_shift", "abs_tz_shift"]:
        team_games_roll[col] = team_games_roll[col].fillna(0)
    team_games_roll["starters_out"] = team_games_roll["starters_out"].fillna(0)

    if pro_bowlers is None:
        from . import data
        pro_bowlers = data.load_pro_bowlers()
    team_games_roll = add_pb_features(
        team_games_roll, injuries, depth_charts, pro_bowlers
    )
    console.info(
        f"pro bowl: mean pb_out_ever={team_games_roll['pb_out_ever'].mean():.3f}, "
        f"pb_out_elite={team_games_roll['pb_out_elite'].mean():.3f}; "
        f"{int((team_games_roll['pb_net_ever'] != 0).sum())} team-games with "
        f"offseason PB churn"
    )

    console.info(
        f"schedule: {int(team_games_roll['is_off_bye'].sum())} off-bye, "
        f"{int(team_games_roll['is_short_week'].sum())} short-week, "
        f"{int(team_games_roll['neutral_site'].sum())} neutral-site team-games; "
        f"mean starters_out={team_games_roll['starters_out'].mean():.2f}"
    )
    return team_games_roll
