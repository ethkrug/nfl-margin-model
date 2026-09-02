"""Backend prediction path: assemble features (incl. projected upcoming games),
train on completed seasons, and return per-game predictions.

This is the single source of truth for producing predictions. The frontend
consumes :func:`predict_games` and only adds presentation (team logos/colours,
JSON, HTML); nothing here knows about the web page.
"""
from __future__ import annotations

import math
import os
from datetime import date

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from . import (advanced, config, console, data, features, fetch, model,
               projection, qb, schedule, weather)

# The model trains on every season <= TRAIN_THROUGH; the season(s) after it are
# the pure hold-out shown to users. Bump each year (2025 -> pick 2026, ...).
TRAIN_THROUGH = 2025
PROJECT_SEASON = TRAIN_THROUGH + 1

# nflverse consensus close (schedules.spread_line), NOT Pinnacle specifically.
SPREAD_SOURCE = "Market"

DRIVERS = {  # frame column -> human label (value > 0 favours the home team)
    "edge_net_epa_roll5": "Net EPA edge (last 5 games)",
    "edge_qb_quality_5": "QB quality edge",
    "edge_off_epa_roll5": "Offense edge",
    "edge_def_epa_roll5": "Defense edge",
    "edge_net_pass_epa_roll5": "Passing matchup edge (last 5 games)",
}


# The raw feeds a --cache directory holds, in the order load_raw returns them.
CACHE_FEEDS = ("pbp", "depth", "injuries", "schedules")


def write_cache(path, frames):
    """Snapshot the raw feeds to ``path`` for a later ``--cache`` run.

    Downloading them is the slow, flaky part of every build -- ~15 seasons of
    play-by-play over HTTPS -- so a run that has them in hand may as well leave
    them on disk. Until this existed, ``--cache`` could only be used by someone
    who had assembled the directory by hand.
    """
    os.makedirs(path, exist_ok=True)
    for name, df in zip(CACHE_FEEDS, frames):
        df.to_parquet(os.path.join(path, f"{name}.parquet"), index=False)
    return path


def load_raw(cache=None, project_season=None, save_cache=None):
    """Load pbp/depth/injuries/schedules, from cached parquet or live nflverse.

    Schedules/depth extend to the upcoming season; pbp/injuries only to played
    seasons (the upcoming season's pbp/injuries don't exist yet).

    ``save_cache`` writes what was fetched to that directory, ready for a later
    ``cache=`` run. It is ignored when ``cache`` is set: there is nothing to
    snapshot, the data already came off disk.
    """
    if cache:
        L = lambda n: pd.read_parquet(os.path.join(cache, f"{n}.parquet"))
        return tuple(L(n) for n in CACHE_FEEDS)
    yrs = list(config.PBP_YEARS)
    syrs = yrs + ([project_season] if project_season and project_season not in yrs else [])
    sched = fetch.schedules(syrs)
    # Depth charts span the old weekly schema and the new daily-snapshot schema;
    # unify them (needs the schedule to date-map the new snapshots to weeks).
    depth = data.load_depth_charts_unified(syrs, sched,
                                           optional=[project_season] if project_season else [])
    frames = (fetch.pbp(yrs), depth, fetch.injuries(yrs), sched)
    if save_cache:
        console.info(f"cache: wrote {', '.join(CACHE_FEEDS)} parquet to "
                     f"{write_cache(save_cache, frames)}")
    return frames


def build_frame(pbp, depth, inj, sched, project_season=None,
                train_through=TRAIN_THROUGH, ref_date=None,
                prior_weight=config.roll_prior_weight, seed_shrink=1.0):
    """Assemble the model frame, optionally appending the next unplayed week.

    Returns ``(tgr, game_df, projected_week)``. When ``project_season`` is set,
    shell rows for the earliest unplayed week are appended and seeded off prior
    games; their QB and weather are supplied by :mod:`projection`/:mod:`weather`.
    """
    if ref_date is None:
        ref_date = date.today()
    ce, cw = data.epa_wpa_columns(pbp)
    gdf = features.build_game_df(features.build_game_totals(pbp, ce, cw))
    tg = features.build_team_games(gdf, features.build_team_game_sums(pbp))
    adv = advanced.build_advanced_metrics(pbp)
    tg = tg.merge(adv, on=["game_id", "team"], how="left")
    ac = [c for c in adv.columns if c not in ("game_id", "team")]
    tg[ac] = tg[ac].fillna(tg[ac].mean())
    precip = advanced.build_precip(pbp)

    proj_week = None
    if project_season is not None:
        clim = weather.month_climatology(gdf, sched, precip)
        ctx, shell, proj_week = projection.projected_frames(
            sched, project_season, gdf, tg, clim, ref_date)
        if ctx is not None:
            precip = pd.concat([precip, ctx[["game_id", "precip"]]], ignore_index=True)
            gdf = pd.concat([gdf, ctx], ignore_index=True)
            tg = pd.concat([tg, shell], ignore_index=True)
            # add_targets aligns team_games to the rolled frame positionally, so
            # keep tg in the (team, season, week) order the roller produces.
            tg = tg.sort_values(["team", "season", "week"]).reset_index(drop=True)

    tgr = features.build_rolling_features(tg, adjust_defense=True,
                                          prior_weight=prior_weight, seed_shrink=seed_shrink)
    tgr = features.add_game_context(tgr, gdf, precip)
    tgr, replacement = qb.add_qb_features(tgr, pbp, depth, inj)
    if project_season is not None and proj_week is not None:
        tgr = projection.override_projected_qb(
            tgr, project_season, train_through, replacement, depth=depth)
    tgr = schedule.add_schedule_features(tgr, sched, depth, inj)
    tgr = features.add_matchup_features(tgr)
    tgr, _ = model.add_targets(tgr, tg, gdf)
    return tgr[~tgr["drop_for_model"]].copy().reset_index(drop=True), gdf, proj_week


def predict_games(cache=None, generated="today", train_through=TRAIN_THROUGH,
                  project_season=None, save_cache=None):
    """Train on completed seasons and return ``(records, meta)``.

    ``records`` is a list of per-game prediction dicts (home perspective);
    ``meta`` carries display config. No presentation concerns here.
    """
    project_season = project_season if project_season is not None else train_through + 1
    try:
        ref_date = date.fromisoformat(str(generated)[:10])
    except Exception:
        ref_date = date.today()

    pbp, depth, inj, sched = load_raw(cache, project_season=project_season,
                                      save_cache=save_cache)
    tgr, gdf, proj_week = build_frame(pbp, depth, inj, sched,
                                      project_season=project_season,
                                      train_through=train_through, ref_date=ref_date)

    available = sorted(int(s) for s in tgr["season"].unique())
    display_seasons = [s for s in available if s > train_through] or [available[-1]]

    drop = [c for c in config.DROP_BEFORE_MODEL if c in tgr.columns]
    feats = [c for c in tgr.columns
             if c not in drop + ["y_margin", "y_win", "drop_for_model"]]
    tr = tgr[tgr["season"] <= train_through]
    clf = Pipeline([("prep", model.build_preprocessor()),
                    ("xgb", XGBRegressor(**config.XGB_PARAMS))])
    clf.fit(tr[feats], tr["y_margin"])
    # Pro Bowl absence correction: slope fit out-of-fold on training seasons
    # only, then applied to every row (a no-op where no differential exists).
    pb_corr = model.fit_pb_correction(tr, feats)
    raw_pred = clf.predict(tgr[feats])
    tgr["pred_raw"] = raw_pred
    tgr["pred"] = model.apply_pb_correction(raw_pred, tgr, pb_corr)

    drivers = {k: v for k, v in DRIVERS.items() if k in tgr.columns}
    sd = float((tr["y_margin"] ** 2).mean() ** 0.5)
    spread_map = (sched.dropna(subset=["spread_line"])
                  .set_index("game_id")["spread_line"].to_dict())
    info_map = sched.set_index("game_id")[
        [c for c in ["weekday", "gametime", "stadium", "location"]
         if c in sched.columns]
    ].to_dict("index")

    def _margin_to_prob(margin):
        """Map a point margin to a home win probability (shared scale)."""
        return 1.0 / (1.0 + math.exp(-margin / (sd * 0.55)))

    def _kickoff(info):
        """'Sunday' + '13:00' -> 'Sun 1:00 PM'."""
        day = str(info.get("weekday") or "")[:3]
        raw = str(info.get("gametime") or "")
        try:
            hh, mm = (int(x) for x in raw.split(":")[:2])
        except (ValueError, TypeError):
            return day or None
        suffix = "AM" if hh < 12 else "PM"
        hour = hh % 12 or 12
        return f"{day} {hour}:{mm:02d} {suffix}".strip()

    home = tgr[tgr["is_home"] == 1].set_index("game_id")
    away = tgr[tgr["is_home"] == 0].set_index("game_id")
    records = []
    for gid in home.index:
        if gid not in away.index:
            continue
        h, a = home.loc[gid], away.loc[gid]
        if int(h["season"]) not in display_seasons:
            continue
        pred = float((h["pred"] - a["pred"]) / 2.0)   # antisymmetric average
        # The pick comes from this averaged margin, so the no-flip guarantee has
        # to hold here too (per-row gating cannot ensure it survives averaging).
        base = float((h["pred_raw"] - a["pred_raw"]) / 2.0)
        if config.PB_CORRECTION_NO_FLIP and base != 0.0 and (pred > 0) != (base > 0):
            pred = base
        hs = None if pd.isna(h["points_for"]) else int(h["points_for"])
        as_ = None if pd.isna(h["points_against"]) else int(h["points_against"])
        wp = _margin_to_prob(pred)
        mkt = spread_map.get(gid)
        mkt = None if mkt is None or pd.isna(mkt) else round(float(mkt), 1)
        # Market-implied home win probability, on the SAME margin->probability
        # scale as the model's own, so the two are directly comparable (a
        # moneyline-derived number would just echo the model back at itself).
        mkt_wp = None if mkt is None else round(_margin_to_prob(mkt), 3)
        info = info_map.get(gid, {})
        venue = str(info.get("stadium") or "") or None
        if venue and str(info.get("location", "Home")) == "Neutral":
            venue += " · neutral site"
        records.append(dict(
            game_id=gid, season=int(h["season"]), week=int(h["week"]),
            home=h["team"], away=h["opponent"],
            neutral_site=int(h.get("neutral_site", 0) or 0),
            is_playoff=int(str(h.get("season_type", "REG")) == "POST"),
            pred_margin=round(pred, 1), win_prob_home=round(wp, 3),
            market_spread=mkt,
            market_win_prob_home=mkt_wp,
            spread_edge=(None if mkt is None else round(pred - mkt, 1)),
            kickoff=_kickoff(info), venue=venue,
            home_score=hs, away_score=as_,
            actual_margin=(None if hs is None else hs - as_),
            drivers={lbl: (None if pd.isna(h[c]) else round(float(h[c]), 3))
                     for c, lbl in drivers.items()},
        ))

    records.sort(key=lambda g: (g["season"], g["week"], g["home"]))
    meta = dict(generated=generated,
                model="Opp-adjusted EPA · matchup · QB · schedule (XGBoost)",
                train_through=train_through, display_seasons=display_seasons,
                project_season=project_season, projected_week=proj_week,
                spread_source=SPREAD_SOURCE, driver_keys=list(drivers.values()))
    return records, meta
