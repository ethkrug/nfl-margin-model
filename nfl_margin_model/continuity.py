"""Continuity-weighted decay averages of each team's per-game stats.

Replaces the 10-game rolling window with one slow-decaying average per stat that
runs across season boundaries but carries the previous seasons over according to
how much of the team is still the same.

For team-game ``i`` and each stat, over the team's earlier games ``j`` from this
season and the previous ``lookback_seasons`` seasons::

    v_i = C_i + sum_j w_ij * G_ij * R_ij * (x_j - C_ij) / sum_j w_ij

* ``w_ij`` -- a linear fade in games back: ``1 - (k - 1) / fade_games``, so the
  previous game counts 1, then 0.97, 0.94, ... (about 0.53 one season back).
* ``C_i`` -- the league level of the stat this season so far (weeks before ``i``'s
  week; last season's level in week 1). Stats are carried relative to the
  league, so a rule change that moves a whole stat (the 2024 and 2025 kickoff
  rules took touchbacks from ~3.5 to ~1.3 a game) does not leak across seasons.
  ``C_ij`` is ``C_i`` for this season's games and the league level of game
  ``j``'s own season for earlier ones.
* ``R_ij`` -- how much that stat carries from one season to the next
  (``rho ** seasons crossed``), measured from team-season means of seasons
  before ``i``'s season only.
* ``G_ij`` -- how that carryover changes with continuity, also measured from
  seasons before ``i``'s season only, from offense/defense/net EPA per play:
  offense and net stats by (same QB?, same head coach?), defense and other stats
  by head coach. Within a season, a game played by a different QB than this
  week's expected starter is shrunk by the new-QB/kept-QB carryover ratio.

The QB for game ``i`` is the pre-game expected starter
(:func:`qb.pregame_starters`: depth chart minus ruled-out QBs); the QB for an
earlier game is whoever took the most dropbacks in it; the coach is that game's
head coach from the schedule. Every input is known before ``i`` kicks off.

Measured on 2011-2023 (EPA per play, season to season), last season's offense
carries over 0.46 when the QB and head coach stayed and -0.27 when both changed;
defense carries ~0.17-0.30 regardless of the QB. 36% of team-seasons have a new
primary QB and 24% a new head coach.

Selected on the 2014-2023 holdouts (each trained on every earlier season, 5
seeds): 13.256 RMSE vs 13.305 for the 5+10-game windows. A plain slow decay
without continuity scored 13.244 there but 12.85 on 2024/2025 and 12.55 on
2026 weeks 1-8, against the windows' 12.73 / 12.45: carryover of team strength
fell from ~0.45-0.52 (2018-23) to ~0.28-0.33 (2024-26), and an uncorrected
long memory misfires when teams turn over. This version scored 12.78 on
2024/2025 and 12.38 on 2026 weeks 1-8. Extra offseason down-weighting
(0.7 / 0.4 per season boundary, or 0.5 when the QB or coach changed) was worse
than shrinking the carried level by continuity, so there is none.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, data
from . import qb as qbm

# Counting stats (summed over the team's own possessions) that describe its
# offense; every other non-prefixed stat is treated as special teams / other.
OFF_COUNT = {
    "pass_attempt", "complete_pass", "rush_attempt", "interception", "sack", "qb_hit",
    "touchdown", "pass_touchdown", "rush_touchdown", "third_down_converted",
    "third_down_failed", "fourth_down_converted", "fourth_down_failed", "fumble",
    "fumble_lost", "tackled_for_loss", "two_point_attempt", "extra_point_attempt",
    "field_goal_attempt", "penalty",
}
QC = [(q, c) for q in (True, False) for c in (True, False)]   # (same QB?, same coach?)


def side_of(col):
    """Which continuity a stat follows: 'off', 'def', 'net' or 'oth'.

    The ``for_``/``allow_`` EPA and WPA columns are built from nflverse's
    zero-sum ``total_home_*``/``total_away_*`` columns, so each is the team's
    NET value (``allow_ == -for_``) and follows net continuity. If those are ever
    rebuilt as true offense/defense values from the plays, map ``for_`` to
    'off' and ``allow_`` to 'def' here.
    """
    if col.startswith(("for_", "allow_")):
        return "net"
    if col.startswith("off_") or col in OFF_COUNT:
        return "off"
    if col.startswith("def_"):
        return "def"
    return "oth"


def inputs(pbp, depth, injuries, sched):
    """Everything the decay needs beyond the team-game stats, from pre-game feeds.

    Returns ``{"games": ..., "team_epa": ...}``: per (game_id, team) the head coach,
    the expected starter (``qb_exp``) and the QB who actually played (``qb_act``,
    empty for unplayed games); and per (team, season) offense/defense/net EPA
    per play, used only to measure how carryover depends on continuity.
    """
    cmap = data.team_code_map()
    d = pbp[pbp["qb_dropback"].eq(1) & pbp["passer_player_id"].notna() & pbp["posteam"].notna()]
    qb_game = (d.groupby(["game_id", "posteam", "passer_player_id"]).size()
               .rename("qb_dropbacks").reset_index())
    qb_game["posteam"] = qb_game["posteam"].replace(cmap)
    actual = (qb_game.sort_values(["game_id", "posteam", "qb_dropbacks", "passer_player_id"],
                                  ascending=[True, True, False, True], kind="mergesort")
              .groupby(["game_id", "posteam"], as_index=False).first()
              .rename(columns={"posteam": "team", "passer_player_id": "qb_act"})
              [["game_id", "team", "qb_act"]])

    s = sched.dropna(subset=["season", "week"])
    cols = ["game_id", "season", "week", "team", "coach"]
    keys = pd.concat([
        s[["game_id", "season", "week", "home_team", "home_coach"]].set_axis(cols, axis=1),
        s[["game_id", "season", "week", "away_team", "away_coach"]].set_axis(cols, axis=1),
    ], ignore_index=True)
    keys["team"] = keys["team"].replace(cmap)
    keys["season"] = keys["season"].astype(int)
    keys["week"] = keys["week"].astype(int)
    qb_depth, _ = qbm._build_qb_depth(depth)
    pre = qbm.pregame_starters(keys[["game_id", "season", "week", "team"]], qb_depth, injuries, qb_game)
    games = (keys.merge(pre[["game_id", "team", "qb_pregame_id"]], on=["game_id", "team"], how="left")
             .merge(actual, on=["game_id", "team"], how="left")
             .rename(columns={"qb_pregame_id": "qb_exp"})
             .drop_duplicates(["game_id", "team"]))

    q = pbp[pbp["play_type"].isin(["pass", "run"]) & pbp["epa"].notna() & pbp["posteam"].notna()
            & (pbp["season_type"] == "REG")][["season", "posteam", "defteam", "epa"]].copy()
    q["posteam"] = q["posteam"].replace(cmap)
    q["defteam"] = q["defteam"].replace(cmap)
    off = q.groupby(["posteam", "season"])["epa"].mean().rename_axis(["team", "season"])
    dfn = q.groupby(["defteam", "season"])["epa"].mean().rename_axis(["team", "season"])
    team_epa = pd.DataFrame({"off": off, "def": dfn})
    team_epa["net"] = team_epa["off"] - team_epa["def"]
    return {"games": games, "team_epa": team_epa}


def _league_levels(X, ok, season, week, stat_cols, tg):
    """Full-season league means, and each row's season-so-far level (weeks before it)."""
    LS = tg.groupby("season")[stat_cols].mean()
    sofar = np.full(X.shape, np.nan)
    for s in np.unique(season):
        m = season == s
        weeks = np.unique(week[m])
        sums = np.vstack([np.nansum(X[m & (week == w)], axis=0) for w in weeks])
        cnts = np.vstack([ok[m & (week == w)].sum(axis=0) for w in weeks])
        before_s, before_c = np.cumsum(sums, axis=0) - sums, np.cumsum(cnts, axis=0) - cnts
        prev = LS.loc[s - 1].to_numpy() if (s - 1) in LS.index else np.full(len(stat_cols), np.nan)
        for wi, w in enumerate(weeks):
            sofar[m & (week == w)] = np.where(before_c[wi] > 0,
                                              before_s[wi] / np.maximum(before_c[wi], 1), prev)
    return LS, sofar


def _stat_carryover(tg, stat_cols, seasons):
    """Per-stat season-to-season carryover, using only season pairs before each season."""
    p = config.CONTINUITY_DECAY
    TS = tg.groupby(["team", "season"])[stat_cols].mean()
    TSd = TS - TS.groupby(level="season").transform("mean")
    num, den = {}, {}
    for t in seasons:
        if t - 1 not in seasons:
            continue
        a, b = TSd.xs(t, level="season"), TSd.xs(t - 1, level="season")
        j = a.index.intersection(b.index)
        num[t] = np.nansum(a.loc[j].to_numpy() * b.loc[j].to_numpy(), axis=0)
        den[t] = np.nansum(b.loc[j].to_numpy() ** 2, axis=0)
    rho = {}
    for s in seasons:
        ts = [t for t in num if t < s]
        if not ts:
            rho[s] = np.full(len(stat_cols), 0.5)
            continue
        nn, dd = sum(num[t] for t in ts), sum(den[t] for t in ts)
        rho[s] = np.clip(np.where(dd > 0, nn / np.where(dd > 0, dd, 1), 0.5), 0, p["rho_max"])
    return rho


def _continuity_multipliers(team_epa, meta, seasons):
    """How carryover of offense/defense/net EPA per play changes with QB and coach
    continuity, relative to all teams -- from season pairs before each season."""
    p = config.CONTINUITY_DECAY
    E = team_epa.join(meta, how="inner")
    Ed = E[["off", "def", "net"]] - E[["off", "def", "net"]].groupby(level="season").transform("mean")
    rows = []
    for team, t in E.index:
        if (team, t - 1) not in E.index:
            continue
        a, b = E.loc[(team, t)], E.loc[(team, t - 1)]
        rows.append(dict(t=t, same_qb=a["qb"] == b["qb"], same_coach=a["coach"] == b["coach"],
                         **{f"{x}_a": Ed.at[(team, t), x] for x in ("off", "def", "net")},
                         **{f"{x}_b": Ed.at[(team, t - 1), x] for x in ("off", "def", "net")}))
    P = pd.DataFrame(rows)

    def mult(sub, all_, side):
        slope = lambda d: (d[f"{side}_a"] * d[f"{side}_b"]).sum() / max((d[f"{side}_b"] ** 2).sum(), 1e-12)
        base = slope(all_)
        if len(sub) == 0 or base <= 0:
            return 1.0
        m = float(np.clip(slope(sub) / base, 0, p["mult_max"]))
        return (len(sub) * m + p["prior_team_seasons"]) / (len(sub) + p["prior_team_seasons"])

    out = {}
    for s in seasons:
        d = P[P["t"] < s] if len(P) else P
        if len(d) < 30:   # too few season pairs to measure anything: no adjustment
            out[s] = dict(off={g: 1.0 for g in QC}, net={g: 1.0 for g in QC},
                          dfn={True: 1.0, False: 1.0}, qmid_off=1.0, qmid_net=1.0)
            continue
        grp = {g: d[(d["same_qb"] == g[0]) & (d["same_coach"] == g[1])] for g in QC}
        off = {g: mult(grp[g], d, "off") for g in QC}
        net = {g: mult(grp[g], d, "net") for g in QC}
        dfn = {c: mult(d[d["same_coach"] == c], d, "def") for c in (True, False)}
        out[s] = dict(off=off, net=net, dfn=dfn,
                      qmid_off=float(np.clip(off[(False, True)] / max(off[(True, True)], 1e-6), 0, 1)),
                      qmid_net=float(np.clip(net[(False, True)] / max(net[(True, True)], 1e-6), 0, 1)))
    return out


def add_decay(tg, stat_cols, cont, suffix):
    """Return ``tg`` (sorted by team, season, week) with ``<stat>_roll<suffix>`` columns."""
    p = config.CONTINUITY_DECAY
    cmap = data.team_code_map()
    tg = tg.sort_values(["team", "season", "week"]).reset_index(drop=True)
    games = tg[["game_id"]].assign(team=tg["team"].replace(cmap)).merge(
        cont["games"], on=["game_id", "team"], how="left")
    if len(games) != len(tg):
        raise ValueError("continuity inputs are not unique per (game_id, team)")
    coach = games["coach"].to_numpy(object)
    qexp, qact = games["qb_exp"].to_numpy(object), games["qb_act"].to_numpy(object)
    season, week = tg["season"].to_numpy(int), tg["week"].to_numpy(int)
    X = tg[stat_cols].to_numpy(float)
    ok = ~np.isnan(X)
    X0 = np.where(ok, X, 0.0)
    seasons = np.unique(season)
    sides = np.array([side_of(c) for c in stat_cols])

    LS, sofar = _league_levels(X, ok, season, week, stat_cols, tg)
    dev_own = np.where(ok, X - LS.loc[season].to_numpy(), 0.0)   # vs own season's level
    rho = _stat_carryover(tg, stat_cols, seasons)

    def mode(sr):
        sr = sr.dropna()
        return sr.value_counts().index[0] if len(sr) else None
    meta = (pd.DataFrame({"team": games["team"].to_numpy(), "season": season, "coach": coach, "qb": qact})
            .groupby(["team", "season"]).agg(coach=("coach", mode), qb=("qb", mode)))
    MULT = _continuity_multipliers(cont["team_epa"], meta, seasons)

    out = np.full(X.shape, np.nan)
    for _, g in tg.groupby("team", sort=False):
        idx = g.index.to_numpy()
        n = len(idx)
        si = season[idx]
        I, J = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        k = (I - J).astype(float)
        gap = si[I] - si[J]
        valid = (k >= 1) & (gap >= 0) & (gap <= p["lookback_seasons"])
        w = np.where(valid, np.clip(1 - (np.maximum(k, 1) - 1) / p["fade_games"], 0, None), 0.0)
        co, qe, qa = coach[idx], qexp[idx], qact[idx]
        same_coach = (co[I] == co[J]) | pd.isna(co[I]) | pd.isna(co[J])
        same_qb = (qe[I] == qa[J]) | pd.isna(qe[I]) | pd.isna(qa[J])
        cross = gap >= 1

        G = {sd: np.ones((n, n)) for sd in ("off", "net", "def")}
        for r in range(n):
            M = MULT[si[r]]
            for sd in ("off", "net"):
                G[sd][r] = np.where(cross[r],
                                    [M[sd][(bool(q), bool(c))] for q, c in zip(same_qb[r], same_coach[r])],
                                    np.where(same_qb[r], 1.0, M[f"qmid_{sd}"]))
            G["def"][r] = np.where(cross[r], [M["dfn"][bool(c)] for c in same_coach[r]], 1.0)
        G["oth"] = G["def"]

        Ci = sofar[idx]
        rr = np.vstack([rho[s] for s in si])
        for sd in ("off", "net", "def", "oth"):
            cols = np.where(sides == sd)[0]
            if not len(cols):
                continue
            okc = ok[idx][:, cols].astype(float)
            den = w @ okc
            wg = w * G[sd]
            same = np.where(cross, 0.0, wg)
            Xc, Dc = X0[idx][:, cols], dev_own[idx][:, cols]
            num = same @ Xc - (same @ okc) * Ci[:, cols]
            for back in range(1, p["lookback_seasons"] + 1):
                num = num + rr[:, cols] ** back * (np.where(gap == back, wg, 0.0) @ Dc)
            v = Ci[:, cols] + num / np.where(den > 0, den, 1)
            out[np.ix_(idx, cols)] = np.where(den > 0, v, np.nan)

    new = pd.DataFrame(out, columns=[f"{c}_roll{suffix}" for c in stat_cols], index=tg.index)
    return pd.concat([tg, new], axis=1)
