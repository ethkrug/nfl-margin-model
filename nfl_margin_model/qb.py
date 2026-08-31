"""Starting-QB features: depth-chart role + shrinkage QB quality.

The team offensive-EPA features are driven mostly by whoever plays QB, but a
*team* average hides who actually started and how good they are. The old design
inferred "backup" purely from a start count (``prior_starts < 4``), which
conflated inexperience with role: it mislabeled rookie franchise starters, let
long-time journeyman backups "graduate" to full starter quality after 4 spot
starts, and gave no signal when the real starter was injured.

This module separates the two questions:

* **Role** -- is the passer the team's *designated* QB1, or pressed into duty?
  Answered from weekly **depth charts** (``qb_depth_order``,
  ``is_designated_starter``) and **injury reports** (``starter_inactive``).
* **Quality** -- how good is this QB relative to expectation, without an
  artificial cliff at 4 starts? Answered with a **shrinkage** estimate
  (``qb_quality_5``, ``qb_quality_10``) that blends a QB's own leakage-safe
  rolling EPA toward replacement level by how many prior starts they have.

No hard-coded penalty: the model is given the role/quality features and learns
the effect itself.

Resulting model features: ``qb_quality_5``, ``qb_quality_10``,
``qb_depth_order``, ``is_designated_starter``, ``starter_inactive``, and
``qb_prior_starts``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, console, data


def _coalesce(df, *candidates):
    """Return the first present candidate column, filled by later ones.

    nflverse depth charts ship in two layouts across seasons (old:
    ``club_code``/``depth_team``/``position``; new: ``team``/``pos_rank``/
    ``pos_abb``). This coalesces whichever are present so either layout works.
    """
    out = None
    for name in candidates:
        if name in df.columns:
            col = df[name]
            out = col if out is None else out.fillna(col)
    if out is None:
        out = pd.Series(np.nan, index=df.index)
    return out


def _normalize_depth(depth_charts):
    """Flatten the multi-schema depth charts into stable, typed columns."""
    out = pd.DataFrame(index=depth_charts.index)
    out["season"] = pd.to_numeric(_coalesce(depth_charts, "season"), errors="coerce")
    out["week"] = pd.to_numeric(_coalesce(depth_charts, "week"), errors="coerce")
    # Historical club codes (OAK/SD/STL) must be mapped to the franchise's current
    # code before ANY join on team; see data.team_code_map for why and how. This
    # single point fixes qb.py, schedule.py and projection.py together.
    out["team"] = _coalesce(depth_charts, "club_code", "team").replace(
        data.team_code_map()
    )
    out["player_id"] = _coalesce(depth_charts, "gsis_id")
    out["position"] = _coalesce(depth_charts, "position", "pos_abb")
    out["formation"] = _coalesce(depth_charts, "formation")
    out["depth_rank"] = pd.to_numeric(
        _coalesce(depth_charts, "depth_team", "pos_rank"), errors="coerce"
    )
    return out


def _build_qb_depth(depth_charts):
    """Normalize depth charts into per (season, week, team, player) QB rank.

    Returns ``(qb_depth, qb1)`` where ``qb_depth`` carries each QB's depth rank
    and ``qb1`` is the designated starter (lowest rank) per team-week. Rows
    without a usable season/week (e.g. an alternate-schema dump) are dropped,
    so those team-games fall back to neutral role features downstream.
    """
    dc = _normalize_depth(depth_charts)
    qb_depth = dc[(dc["position"] == "QB") & (dc["formation"] == "Offense")].copy()
    qb_depth = qb_depth.dropna(subset=["season", "week", "depth_rank", "player_id"])
    qb_depth["season"] = qb_depth["season"].astype(int)
    qb_depth["week"] = qb_depth["week"].astype(int)

    qb_depth = qb_depth[["season", "week", "team", "player_id", "depth_rank"]]
    # Ties on depth_rank are common -- 155 of ~9,000 team-weeks list two or more
    # QBs at the same rank -- so ``depth_rank`` alone is not a total order, and
    # pandas' default quicksort is NOT stable. Sorting on rank alone therefore
    # let the winner of a tie depend on the surrounding array layout: adding or
    # removing unrelated rows elsewhere in the frame silently reassigned QB1 for
    # team-weeks in other seasons entirely. ``player_id`` makes the order total,
    # and mergesort keeps it stable, so the result depends only on the rows in
    # each team-week.
    sort_key = ["depth_rank", "player_id"]
    qb_depth = (
        qb_depth.sort_values(sort_key, kind="mergesort")
        .drop_duplicates(["season", "week", "team", "player_id"], keep="first")
    )

    # Designated starter = the QB with the lowest depth rank that week.
    qb1 = (
        qb_depth.sort_values(sort_key, kind="mergesort")
        .groupby(["season", "week", "team"], as_index=False)
        .first()[["season", "week", "team", "player_id"]]
        .rename(columns={"player_id": "qb1_id"})
    )
    return qb_depth, qb1


def _build_role_features(starters, qb_depth, qb1, injuries):
    """Per team-game role features from depth charts + injuries.

    ``qb_depth``/``qb1`` come from :func:`_build_qb_depth`; the caller computes
    them once because the starter selection needs ``qb1`` too.

    Returns a frame keyed on ``["game_id", "team"]`` with ``qb_depth_order``,
    ``is_designated_starter`` and ``starter_inactive``.
    """

    s = starters[["game_id", "season", "week", "posteam", "passer_player_id"]].rename(
        columns={"posteam": "team", "passer_player_id": "player_id"}
    )

    # Depth rank of the QB who actually started.
    s = s.merge(
        qb_depth[["season", "week", "team", "player_id", "depth_rank"]],
        on=["season", "week", "team", "player_id"],
        how="left",
    )
    s = s.merge(qb1, on=["season", "week", "team"], how="left")

    # Injury status of the designated QB1 that week.
    inj = injuries[["season", "week", "team", "gsis_id", "report_status"]].copy()
    inj["season"] = pd.to_numeric(inj["season"], errors="coerce")
    inj["week"] = pd.to_numeric(inj["week"], errors="coerce")
    inj = inj.dropna(subset=["season", "week"])
    inj["season"] = inj["season"].astype(int)
    inj["week"] = inj["week"].astype(int)
    inj = inj.rename(columns={"gsis_id": "qb1_id"})
    # A player can appear twice in a week with conflicting report statuses (2
    # such cases in 2010-2025, both disagreeing on whether he was out). Keeping
    # whichever row came first made that an arbitrary, layout-dependent call, so
    # rank by severity and keep the most severe: if any report rules the starter
    # out, he is out.
    inj["_inactive"] = inj["report_status"].isin(config.QB_INACTIVE_STATUSES).astype(int)
    inj = (
        inj.sort_values(
            ["season", "week", "team", "qb1_id", "_inactive"],
            ascending=[True, True, True, True, False],
            kind="mergesort",
        )
        .drop_duplicates(["season", "week", "team", "qb1_id"], keep="first")
        .drop(columns="_inactive")
    )

    s = s.merge(inj, on=["season", "week", "team", "qb1_id"], how="left")

    # qb_depth_order: passer not located on the QB chart -> deep-backup sentinel.
    s["qb_depth_order"] = s["depth_rank"].fillna(config.QB_DEPTH_SENTINEL).astype(int)

    # is_designated_starter: when the team-week has no depth data (qb1 unknown),
    # assume a normal start (1) rather than wrongly flagging a non-starter.
    s["is_designated_starter"] = np.where(
        s["qb1_id"].isna(),
        1,
        (s["player_id"] == s["qb1_id"]).astype(int),
    ).astype(int)

    s["starter_inactive"] = (
        s["report_status"].isin(config.QB_INACTIVE_STATUSES).astype(int)
    )

    return s[["game_id", "team", "qb_depth_order", "is_designated_starter", "starter_inactive"]]


def add_qb_features(team_games_roll, play_by_play_df, depth_charts, injuries):
    """Compute starting-QB features and merge them onto ``team_games_roll``.

    Returns ``(team_games_roll, replacement_epa)``.
    """
    windows = config.QB_WINDOWS

    # 1) QB dropback plays -- pass attempts + sacks carry an identifiable passer.
    #    qb_epa credits the QB for the play's EPA; fall back to epa if missing.
    qb_plays = play_by_play_df.loc[
        play_by_play_df["passer_player_id"].notna()
        & play_by_play_df["posteam"].notna()
        & play_by_play_df["qb_dropback"].fillna(0).eq(1)
    ].copy()
    qb_plays["qb_epa_use"] = qb_plays["qb_epa"].fillna(qb_plays["epa"])

    # 2) one row per QB per game: EPA / dropback + dropback count.
    qb_game = (
        qb_plays.groupby(
            ["game_id", "season", "week", "posteam", "passer_player_id"],
            as_index=False,
        ).agg(
            passer_player_name=("passer_player_name", "first"),
            game_date=("game_date", "first"),
            qb_dropbacks=("qb_epa_use", "size"),
            qb_game_epa=("qb_epa_use", "mean"),
        )
    )
    qb_game["game_date"] = pd.to_datetime(qb_game["game_date"], errors="coerce")

    # 3) starter = QB with the most dropbacks for that team in that game.
    # qb_dropbacks alone is not a total order -- 14 of ~8,700 team-games have two
    # QBs tied on the most dropbacks -- and pandas' default sort is unstable, so
    # the "starter" for those games used to depend on incidental array layout.
    #
    # Break the tie on the evidence rather than arbitrarily: prefer whoever the
    # team published as QB1 on that week's depth chart. That is a pre-game feed,
    # so it is leakage-safe, and it is the same question the tie is asking. Fall
    # back to passer_player_id only when the chart cannot separate them (neither
    # is QB1, or the team-week has no depth data), purely so the result is
    # reproducible. Measured: the chart settles 13 of the 14 ties, 1 falls
    # through, 0 are ambiguous. Same bug class as _build_qb_depth above.
    qb_depth, qb1 = _build_qb_depth(depth_charts)
    qb_game = qb_game.merge(
        qb1.rename(columns={"team": "posteam"}),
        on=["season", "week", "posteam"], how="left",
    )
    qb_game["_is_qb1"] = (
        qb_game["passer_player_id"] == qb_game["qb1_id"]
    ).astype(int)
    starters = (
        qb_game.sort_values(
            ["game_id", "posteam", "qb_dropbacks", "_is_qb1", "passer_player_id"],
            ascending=[True, True, False, False, True],
            kind="mergesort",
        )
        .groupby(["game_id", "posteam"], as_index=False)
        .first()
        .drop(columns=["_is_qb1", "qb1_id"])
    )

    # 4) roll each QB's OWN prior form. shift(1) excludes the current game (no
    #    leak); min_periods=1 lets the feature exist from the QB's 2nd start on.
    def qb_rolls(g, windows=windows):
        g = g.sort_values(["game_date", "game_id"]).copy()
        prior = g["qb_game_epa"].shift(1)
        for w in windows:
            g[f"qb_roll_epa_{w}"] = prior.rolling(w, min_periods=1).mean().to_numpy()
        g["qb_prior_starts"] = np.arange(len(g))  # starts made BEFORE this game
        # Career-to-date baseline (true-talent estimate) and how many starts the
        # QB has already made THIS season -- both leakage-safe (built from prior).
        g["qb_career_epa"] = prior.expanding(min_periods=1).mean().to_numpy()
        g["qb_season_start_idx"] = g.groupby("season").cumcount()
        # Longer-memory rolling form (option 3), leakage-safe.
        g["qb_roll_epa_long"] = prior.rolling(config.QB_LONG_WINDOW, min_periods=1).mean().to_numpy()
        # State AFTER this start, i.e. including it. The representative QB for a
        # game is the one the depth chart named, who may not have played -- his
        # form entering that game is the state left by his own most recent start,
        # which is what these columns carry into the as-of join below.
        inc = g["qb_game_epa"]
        for w in windows:
            g[f"_post_roll_{w}"] = inc.rolling(w, min_periods=1).mean().to_numpy()
        g["_post_starts"] = np.arange(len(g)) + 1
        g["_post_career"] = inc.expanding(min_periods=1).mean().to_numpy()
        g["_post_roll_long"] = inc.rolling(
            config.QB_LONG_WINDOW, min_periods=1).mean().to_numpy()
        g["_post_season"] = g["season"].to_numpy()
        g["_post_season_idx"] = g.groupby("season").cumcount().to_numpy() + 1
        return g

    starters = pd.concat(
        [qb_rolls(g) for _, g in starters.groupby("passer_player_id", sort=False)],
        ignore_index=True,
    )

    # 5) replacement-level form = low percentile of starter EPA (train era only,
    #    so val/test seasons don't leak into the baseline).
    replacement_epa = (
        starters.loc[starters["season"] <= config.TRAIN_MAX_SEASON, "qb_game_epa"]
        .quantile(config.REPLACEMENT_QUANTILE)
    )

    # 5b) THE REPRESENTATIVE QB IS THE ONE THE DEPTH CHART NAMED, NOT THE ONE WHO
    #     TURNED OUT TO PLAY MOST. Which QB ends up taking the most dropbacks is
    #     only knowable after kickoff, so using it to choose whose form describes
    #     the team leaks the game into its own feature. The published depth chart
    #     is a pre-game feed and answers the same question honestly. (Building the
    #     HISTORY from whoever actually threw is not leakage -- those box scores
    #     already exist when the current game kicks off -- so `starters` above,
    #     and the rolling series built from it, stay as they are.)
    #
    #     The named QB1 may not appear in this game at all, so his form is fetched
    #     as-of: the state left by his own most recent start before this kickoff.
    #     Where no QB1 is published, fall back to the QB who actually started.
    post_cols = ([f"_post_roll_{w}" for w in windows]
                 + ["_post_starts", "_post_career", "_post_roll_long",
                    "_post_season", "_post_season_idx"])
    state = (starters[["passer_player_id", "game_date"] + post_cols]
             .dropna(subset=["game_date"])
             .sort_values(["game_date", "passer_player_id"], kind="mergesort"))

    rep = starters[["game_id", "season", "week", "posteam", "game_date",
                    "passer_player_id", "passer_player_name", "qb_dropbacks"]].copy()
    rep = rep.merge(qb1.rename(columns={"team": "posteam"}),
                    on=["season", "week", "posteam"], how="left")
    rep["rep_qb_id"] = rep["qb1_id"].fillna(rep["passer_player_id"])
    rep["qb1_published"] = rep["qb1_id"].notna().astype(int)

    rep = pd.merge_asof(
        rep.sort_values(["game_date", "rep_qb_id"], kind="mergesort"),
        state, on="game_date", left_by="rep_qb_id", right_by="passer_player_id",
        direction="backward", allow_exact_matches=False, suffixes=("", "_state"),
    )
    for w in windows:
        rep[f"qb_roll_epa_{w}"] = rep[f"_post_roll_{w}"]
    rep["qb_prior_starts"] = rep["_post_starts"].fillna(0)
    rep["qb_career_epa"] = rep["_post_career"]
    rep["qb_roll_epa_long"] = rep["_post_roll_long"]
    # Starts already made THIS season; a QB whose last start was a prior season
    # is at 0, which is what triggers the season-opener regression toward career.
    rep["qb_season_start_idx"] = np.where(
        rep["_post_season"].to_numpy() == rep["season"].to_numpy(),
        rep["_post_season_idx"].fillna(0).to_numpy(), 0.0,
    )
    starters = rep

    # 6) shrinkage quality: blend a QB's own rolling form toward replacement by
    #    how many prior starts back the rolling window. A debut (n_eff = 0) maps
    #    exactly to replacement; an established starter keeps their own form; a
    #    long-time backup keeps their genuinely-low form (no graduation cliff).
    K = config.QB_SHRINKAGE_K
    Kc = config.QB_CAREER_SHRINK          # extra shrinkage toward the QB's career mean
    # Offseason regression weight per game: strong at a QB's season opener, fading
    # to 0 as they accumulate current-season starts (see config).
    s = starters["qb_season_start_idx"].to_numpy(dtype=float)
    r = config.QB_OFFSEASON_REG_STRENGTH * np.maximum(0.0, 1.0 - s / config.QB_OFFSEASON_REG_FADE)
    career = starters["qb_career_epa"].fillna(replacement_epa).to_numpy()
    beta = config.QB_LONG_BLEND                         # option 3: long-window blend
    long_form = starters["qb_roll_epa_long"].fillna(replacement_epa)
    for w in windows:
        own_form = starters[f"qb_roll_epa_{w}"].fillna(replacement_epa)
        if beta:
            own_form = (1.0 - beta) * own_form + beta * long_form
        n_eff = np.minimum(starters["qb_prior_starts"].to_numpy(dtype=float), w)
        # Shrink recent form toward the QB's career mean (Kc) and league
        # replacement (K); Kc=0 recovers the original replacement-only shrinkage.
        raw_quality = (n_eff * own_form + Kc * career + K * replacement_epa) / (n_eff + Kc + K)
        raw_quality = raw_quality.to_numpy(dtype=float)
        # Deviation-triggered pull toward career for established QBs only (off by
        # default): surgical fix for elite QBs whose recent form has drifted far.
        if config.QB_DEV_REG_LAMBDA:
            prior_starts = starters["qb_prior_starts"].to_numpy(dtype=float)
            gap = career - raw_quality
            trig = (prior_starts >= config.QB_DEV_MIN_STARTS) & (np.abs(gap) > config.QB_DEV_THRESHOLD)
            raw_quality = raw_quality + np.where(trig, config.QB_DEV_REG_LAMBDA * gap, 0.0)
        # Keep the pre-offseason-pull value. ``override_projected_qb`` seeds the
        # upcoming season from a QB's last row and applies the opener pull itself;
        # if it read the already-pulled qb_quality_* for a QB whose last start WAS
        # an opener, the pull would compound (0.75 -> ~0.94 career weight). That is
        # ~12 of 66 QBs a season, and exactly the ones whose form diverges from
        # career (week-1 injuries), so the projection reads these base columns.
        starters[f"qb_quality_base_{w}"] = raw_quality
        # Optional offseason pull toward career (r=0 by default).
        starters[f"qb_quality_{w}"] = (1.0 - r) * raw_quality + r * career

    # 7) role features from depth charts + injuries (leakage-safe pre-game feeds).
    role = _build_role_features(starters, qb_depth, qb1, injuries)

    # 8) merge starter features onto team_games_roll (one starter per team-game).
    qb_quality_cols = [f"qb_quality_{w}" for w in windows]
    qb_base_cols = [f"qb_quality_base_{w}" for w in windows]
    qb_roll_cols = [f"qb_roll_epa_{w}" for w in windows]
    qb_merge_cols = (
        ["qb_player_id", "qb_player_name", "qb_dropbacks", "qb_prior_starts",
         "qb_career_epa"]
        + qb_roll_cols + qb_quality_cols + qb_base_cols
    )
    # qb_player_id is the QB the model is describing -- the representative
    # (depth-chart) one, not whoever turned out to throw most. projection.py
    # reads this to seed the upcoming season, so the two now agree on what a
    # team's starter means.
    starters = starters.rename(columns={"rep_qb_id": "qb_player_id"})
    name_map = dict(zip(qb_game["passer_player_id"], qb_game["passer_player_name"]))
    starters["qb_player_name"] = starters["qb_player_id"].map(name_map)
    starters_merge = starters.rename(columns={"posteam": "team"})[
        ["game_id", "team"] + qb_merge_cols
    ]

    starters_merge = starters_merge.merge(role, on=["game_id", "team"], how="left")

    team_games_roll = team_games_roll.merge(
        starters_merge, on=["game_id", "team"], how="left"
    )

    # 9) any team-game with no identified starter -> treat as an unknown deep backup.
    team_games_roll["qb_prior_starts"] = team_games_roll["qb_prior_starts"].fillna(0)
    team_games_roll["qb_dropbacks"] = team_games_roll["qb_dropbacks"].fillna(0)
    team_games_roll["qb_depth_order"] = (
        team_games_roll["qb_depth_order"].fillna(config.QB_DEPTH_SENTINEL).astype(int)
    )
    team_games_roll["is_designated_starter"] = (
        team_games_roll["is_designated_starter"].fillna(0).astype(int)
    )
    team_games_roll["starter_inactive"] = (
        team_games_roll["starter_inactive"].fillna(0).astype(int)
    )
    for w in windows:
        team_games_roll[f"qb_roll_epa_{w}"] = team_games_roll[f"qb_roll_epa_{w}"].fillna(
            replacement_epa
        )
        team_games_roll[f"qb_quality_{w}"] = team_games_roll[f"qb_quality_{w}"].fillna(
            replacement_epa
        )
        team_games_roll[f"qb_quality_base_{w}"] = team_games_roll[
            f"qb_quality_base_{w}"
        ].fillna(replacement_epa)

    designated_rate = team_games_roll["is_designated_starter"].mean()
    forced_backups = int(
        (
            (team_games_roll["is_designated_starter"] == 0)
            & (team_games_roll["starter_inactive"] == 1)
        ).sum()
    )
    console.info(
        f"replacement-level EPA/dropback "
        f"({int(config.REPLACEMENT_QUANTILE * 100)}th pct, train era): {replacement_epa:.3f}"
    )
    console.info(
        f"designated QB1 started {designated_rate:.1%} of team-games; "
        f"{forced_backups} injury-forced backup starts "
        f"of {len(team_games_roll)} team-games"
    )
    return team_games_roll, replacement_epa
