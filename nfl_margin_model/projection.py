"""Projected (not-yet-played) games for the current season.

The core pipeline only builds rows for games that have play-by-play. To predict
upcoming games this module appends *shell* team-game rows for the earliest week
that still has games left; the leakage-safe shift(1) rolling then seeds their
features off completed prior games. It also assigns each projected team its
starting QB (from that week's depth chart, falling back to the primary returning
starter).

Nothing here is specific to a season that has not started. As soon as a week is
final its play-by-play is published, so the next build sees it as ordinary
history: the rolling windows advance onto it and the next week is projected off
it, with no config change. What that requires is that the two paths agree --
a projected game must be scored the way a played one is -- which is why the QB
seeding shares :func:`qb.shrink_quality` and
:func:`config.qb_offseason_reg_weight` with the played path rather than
re-deriving them.
"""
from __future__ import annotations

from datetime import timedelta

import pandas as pd

from . import config, qb, weather
from .qb import _normalize_depth


def projected_frames(sched, project_season, gdf, team_games, clim, ref_date):
    """Build game-context + shell team-game rows for the next week with games left.

    Returns ``(context_rows, shell_rows, target_week)`` or ``(None, None, None)``.
    Only ONE week is projected, so its rolling features seed cleanly off
    completed prior games rather than off other projected weeks -- and within it,
    only the games that are actually still to be played.

    "Still to be played" is a property of a GAME, not of its week. An NFL week
    opens with a Thursday night game and finishes on Monday, so a week is
    partially complete for four days out of seven. Retiring the whole week as
    soon as ANY game in it had a score dropped the rest of the slate out of the
    build entirely -- no shell rows, so no features, so no predictions -- and
    jumped the app to next week while 15 games were still upcoming.

    A game is pending when it has no score and is not stale -- see
    :data:`config.PROJECT_STALE_DAYS`. Staleness needs a grace period rather than
    a plain "kickoff is in the past" test, because a game that kicked off last
    night has no score for hours afterwards; without the grace period the build
    would drop a just-played game instead of predicting it, which is the same
    disappearing-game bug in a different place. A game with an unparseable date
    stays pending rather than being silently dropped.
    """
    ps = sched[sched["season"] == project_season]
    if ps.empty:
        return None, None, None
    kickoff = pd.to_datetime(ps["gameday"], errors="coerce").dt.date
    stale = kickoff < ref_date - timedelta(days=config.PROJECT_STALE_DAYS)
    pending = ps[ps["home_score"].isna() & ~stale]
    if pending.empty:
        return None, None, None
    target = int(pending["week"].min())
    wk = pending[pending["week"] == target].copy()

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


def depth_qb1(depth, project_season, max_week=None):
    """Projected-season QB1 per team from nflverse depth charts.

    Empty until nflverse publishes real offseason depth for the upcoming season
    (pre-season it mirrors last year), at which point trades/signings/rookies are
    picked up automatically with no code change.

    ``max_week`` caps which charts may be read, and should be the week being
    projected. Taking the latest chart outright is fine before a season starts,
    when the only charts are week 1's, but mid-season a chart published for a
    LATER week reflects what teams learned after the week being predicted -- so
    reading it would be a small leak as well as simply the wrong starter.
    """
    nd = _normalize_depth(depth)
    nd = nd[(nd["season"] == project_season) & (nd["position"] == "QB")
            & (nd["depth_rank"] == 1)].dropna(subset=["week", "team", "player_id"])
    if max_week is not None:
        nd = nd[nd["week"] <= max_week]
    if nd.empty:
        return {}
    # Several QBs can share depth_rank 1 in a week, so break the tie on
    # player_id rather than on array order.
    latest = nd.sort_values(["week", "player_id"], kind="mergesort").groupby("team").tail(1)
    return dict(zip(latest["team"], latest["player_id"]))


def override_projected_qb(tgr, project_season, train_through, replacement,
                          qb_latest, depth=None):
    """Assign each *unplayed* projected game its starting QB and that QB's quality.

    Starter, in priority order:
      1. the projected week's depth-chart QB1 (nflverse), once published;
      2. else the team's PRIMARY (most-starts) starter from the last completed
         season -- deliberately not most-recent, so an injured-late franchise QB
         (e.g. Mahomes) returns instead of the backup who finished the year.

    Quality comes from ``qb_latest`` -- the QB's form AFTER his most recent start,
    shrunk by :func:`qb.shrink_quality`, the same function that scores a played
    game. A QB with no NFL history (rookie) falls back to replacement level.

    Two things this must get right once the projected season is under way:

    * only rows with no play-by-play are touched. ``qb_player_id`` is set by the
      pbp-derived merge in :func:`qb.add_qb_features`, so a null one marks a
      shell row. Selecting on season alone would overwrite the real, measured QB
      features of games already played in the projected season -- by mid-season,
      nearly all of it.
    * the offseason pull toward career is a function of how many starts the QB
      already has this season (:func:`config.qb_offseason_reg_weight`), not a
      constant. It belongs at a season opener, where there is no current-season
      data and career talent is the better estimate; from week 2 on the QB has
      real current-season form and the pull is off, exactly as for a played game.
    """
    shell = tgr.index[(tgr["season"] == project_season) & tgr["qb_player_id"].isna()]
    if not len(shell):
        return tgr

    target_week = int(tgr.loc[shell, "week"].max())
    played = tgr[(tgr["season"] <= train_through) & tgr["qb_player_id"].notna()]
    if played.empty:
        return tgr
    latest = tgr[(tgr["season"] == train_through) & tgr["qb_player_id"].notna()]
    latest = latest if not latest.empty else played
    counts = latest.groupby(["team", "qb_player_id"]).size().reset_index(name="n")
    # Two QBs can split a season evenly; tie-break on player_id so "primary
    # starter" is reproducible rather than whichever row landed last.
    primary = (counts.sort_values(["n", "qb_player_id"], kind="mergesort")
               .groupby("team").tail(1).set_index("team")["qb_player_id"].to_dict())
    dq1 = (depth_qb1(depth, project_season, max_week=target_week)
           if depth is not None else {})

    for i in shell:
        t = tgr.at[i, "team"]
        pid = dq1.get(t) or primary.get(t)
        if pid is None:
            continue
        if pid in qb_latest.index:
            st = qb_latest.loc[pid]
            starts = float(st["_post_starts"])
            career = st["_post_career"]
            car = replacement if pd.isna(career) else float(career)
            # Starts already made in the projected season: 0 when the QB's last
            # start was a prior season (his opener, so the full pull applies),
            # >= 1 from his second start on (no pull). Mirrors how
            # ``qb_season_start_idx`` is derived for a played game.
            season_starts = (float(st["_post_season_idx"])
                             if int(st["_post_season"]) == int(project_season) else 0.0)
            r0 = float(config.qb_offseason_reg_weight(season_starts))
            for w in config.QB_WINDOWS:
                base = float(qb.shrink_quality(
                    st[f"_post_roll_{w}"], st["_post_roll_long"],
                    career, starts, w, replacement))
                tgr.at[i, f"qb_quality_{w}"] = (1.0 - r0) * base + r0 * car
            if "qb_career_epa" in tgr.columns:
                tgr.at[i, "qb_career_epa"] = career
        else:                              # rookie / no NFL history
            starts = 0.0
            for w in config.QB_WINDOWS:
                tgr.at[i, f"qb_quality_{w}"] = replacement
        tgr.at[i, "qb_prior_starts"] = starts
        tgr.at[i, "qb_depth_order"] = 1
        tgr.at[i, "is_designated_starter"] = 1
        tgr.at[i, "starter_inactive"] = 0
        if "qb_player_id" in tgr.columns:
            tgr.at[i, "qb_player_id"] = pid
    return tgr
