"""Data loading and column discovery."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, console, fetch

# nflverse depth charts changed to a daily-snapshot schema from 2025 on (a ``dt``
# timestamp instead of season/week, with pos_abb/pos_rank instead of
# position/depth_team). This season is the first in the new format.
DEPTH_NEW_FORMAT_FROM = 2025


def _formation_from_posgrp(pos_grp):
    """Map the new format's personnel grouping to Offense/Defense/Special Teams."""
    p = str(pos_grp).lower()
    if "special" in p:
        return "Special Teams"
    if any(k in p for k in (" d", "4-3", "3-4", "nickel", "dime", "def")):
        return "Defense"
    return "Offense"


def _new_depth_to_old(new, schedules):
    """Convert new daily-snapshot depth charts to the old (season, week, ...) schema.

    Each snapshot's date is mapped to the NFL week it precedes (via the schedule's
    kickoffs), so a game's row uses the last snapshot BEFORE that week's kickoff
    (leakage-safe). One row per (season, week, team, player): the latest snapshot
    within the week.
    """
    new = new.copy()
    new["date"] = pd.to_datetime(new["dt"], errors="coerce").dt.tz_localize(None)
    new = new.dropna(subset=["date"])
    m = new["date"].dt.month
    new["season"] = np.where(m >= 3, new["date"].dt.year, new["date"].dt.year - 1)

    sch = schedules.copy()
    sch["gd"] = pd.to_datetime(sch["gameday"], errors="coerce")
    parts = []
    for season, g in new.groupby("season"):
        kk = sch[sch["season"] == season].groupby("week")["gd"].min().dropna().sort_index()
        g = g.copy()
        if len(kk) == 0:
            g["week"] = 1
        else:
            wk = kk.index.values.astype(int)
            kd = kk.values.astype("datetime64[ns]")
            idx = np.searchsorted(kd, g["date"].values.astype("datetime64[ns]"),
                                  side="left")
            # A snapshot maps to the next week that has not kicked off yet.
            # Anything past the final week's kickoff is post-season/offseason and
            # therefore reflects roster state AFTER that season's last games; it
            # is dropped rather than clipped back into the last week, which would
            # hand the final week a chart built from its own future.
            keep = idx < len(wk)
            g = g[keep].copy()
            g["week"] = wk[idx[keep]]
        parts.append(g)
    new = pd.concat(parts, ignore_index=True)
    # Total order + stable sort: ``tail(1)`` must pick the latest snapshot on the
    # row's own merits, never on incidental array layout (see _build_qb_depth).
    new = new.sort_values(["date", "gsis_id"], kind="mergesort").groupby(
        ["season", "week", "team", "gsis_id"], as_index=False).tail(1)

    return pd.DataFrame({
        "season": new["season"].astype(int),
        "week": new["week"].astype(int),
        "club_code": new["team"].astype(str).values,
        # old-format depth_team is a string ("1","2",...); match it so the two
        # schemas concat cleanly (the pipeline coerces to numeric downstream).
        "depth_team": new["pos_rank"].astype("Int64").astype(str).values,
        "position": new["pos_abb"].astype(str).values,
        "gsis_id": new["gsis_id"].values,
        "formation": new["pos_grp"].map(_formation_from_posgrp).values,
    })


# Fallback if team_desc cannot be reached; the live mapping is derived below.
TEAM_CODE_FALLBACK = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA"}
_TEAM_CODE_MAP = None


def team_code_map(schedules=None):
    """Map every historical club code to the franchise's CURRENT code.

    Depth charts carry the code a franchise used at the time (OAK/SD/STL) while
    play-by-play and schedules use the current one (LV/LAC/LA), so an unmapped
    join on team silently misses a decade of games for relocated franchises.

    The mapping is derived rather than hardcoded: nflverse's ``team_desc`` gives
    each franchise a ``team_id`` that survives relocation (the Raiders are 2520
    as both OAK and LV), so every code sharing a ``team_id`` is the same club.
    Which of them is "current" is taken from the schedule itself, so a future
    relocation is picked up with no code change.
    """
    global _TEAM_CODE_MAP
    if _TEAM_CODE_MAP is not None:
        return _TEAM_CODE_MAP
    if schedules is None or not len(schedules):
        # No schedule to date the codes with -- do not guess, use the known map.
        return dict(TEAM_CODE_FALLBACK)
    try:
        td = fetch.team_desc()[["team_abbr", "team_id"]].dropna()
        used = pd.concat([
            schedules[["season", "home_team"]].rename(columns={"home_team": "t"}),
            schedules[["season", "away_team"]].rename(columns={"away_team": "t"}),
        ]).dropna()
        # Canonical = the code the franchise used MOST RECENTLY. Frequency is the
        # wrong test: the schedule itself uses the historical code for historical
        # seasons, so OAK outnumbers LV over 2010-2025 and picking the commonest
        # would map the live code back onto the retired one.
        last_season = used.groupby("t")["season"].max()
        mapping = {}
        for _, g in td.groupby("team_id"):
            abbrs = sorted(set(g["team_abbr"]))
            seen = [a for a in abbrs if a in last_season.index]
            if len(abbrs) < 2 or not seen:
                continue
            canon = max(seen, key=lambda a: last_season[a])
            for a in abbrs:
                if a != canon:
                    mapping[a] = canon
        _TEAM_CODE_MAP = mapping or dict(TEAM_CODE_FALLBACK)
    except Exception as exc:  # offline / schema change -> keep the known aliases
        console.info(f"team code map: falling back to static aliases ({exc})")
        _TEAM_CODE_MAP = dict(TEAM_CODE_FALLBACK)
    return _TEAM_CODE_MAP


def load_depth_charts_unified(years, schedules, optional=()):
    """Load depth charts across ``years`` in either schema, unified to the old one.

    Old-format seasons (< DEPTH_NEW_FORMAT_FROM) load as-is; new daily-snapshot
    seasons are converted via :func:`_new_depth_to_old` and concatenated, so the
    rest of the pipeline sees one consistent (season, week, ...) schema.

    Every requested season must come back with rows, except those named in
    ``optional`` -- the upcoming season, whose charts are not published until it
    exists. A silently absent season blanks the starter and injury-forced-backup
    features for that year, so it is an error rather than a shrug.
    """
    years = list(years)
    team_code_map(schedules)   # warm the cache while a schedule is in hand
    old_years = [y for y in years if y < DEPTH_NEW_FORMAT_FROM]
    new_years = [y for y in years if y >= DEPTH_NEW_FORMAT_FROM]
    frames = []
    if old_years:
        frames.append(fetch.depth_charts(old_years, optional=optional))
    if new_years:
        raw = fetch.depth_charts(new_years, optional=optional)
        if len(raw):
            frames.append(_new_depth_to_old(raw, schedules))
    return pd.concat(frames, ignore_index=True)


def load_data():
    """Import weekly player data and play-by-play data over the configured seasons.

    Returns ``(weekly_df, play_by_play_df)``. The weekly frame is loaded to match
    the original pipeline; only the play-by-play frame drives feature engineering.
    """
    console.step("Importing weekly player data")
    weekly_df = fetch.weekly(list(config.WEEKLY_YEARS), columns=None, downcast=False)
    console.info(f"weekly_df: {len(weekly_df):,} rows")

    console.step("Importing play-by-play data")
    play_by_play_df = fetch.pbp(list(config.PBP_YEARS), columns=None, downcast=False)
    console.info(f"play_by_play_df: {len(play_by_play_df):,} rows")
    return weekly_df, play_by_play_df


def load_depth_charts(schedules=None):
    """Import weekly team depth charts (used to identify the designated QB1).

    Routed through :func:`load_depth_charts_unified` so that seasons on the new
    daily-snapshot schema are date-mapped onto NFL weeks like every other season.
    Reading them raw instead drops those seasons entirely downstream (they carry
    no ``season``/``week``/``formation``), which silently blanks the starter and
    injury-forced-backup features for exactly the most recent seasons.

    ``schedules`` supplies the kickoff dates that mapping needs; it is loaded
    here when the caller has not already got it.
    """
    console.step("Importing depth charts")
    if schedules is None:
        schedules = fetch.schedules(config.PBP_YEARS)
    depth_charts = load_depth_charts_unified(config.PBP_YEARS, schedules)
    console.info(f"depth_charts: {len(depth_charts):,} rows")
    return depth_charts


def load_injuries():
    """Import weekly injury reports (used to flag an unavailable starter)."""
    console.step("Importing injury reports")
    injuries = fetch.injuries(config.PBP_YEARS)
    console.info(f"injuries: {len(injuries):,} rows")
    return injuries


def _pos_family(pos):
    """Collapse a position label to a family for cross-source matching."""
    p = str(pos).upper().strip()
    if p in {"G", "T", "C", "OG", "OT", "OL", "LT", "LG", "RT", "RG"}:
        return "OL"
    return p


def _norm_name(name):
    """Normalize a player name for matching (drop suffixes/punctuation/case)."""
    import re

    s = re.sub(r"[+%*]+", "", str(name)).lower()
    s = re.sub(r"[.'\-]", "", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return re.sub(r"[^a-z]", "", s)


def load_pro_bowlers():
    """Pro Bowl selections per season -> ``{season: set(gsis_id)}``.

    Reads the manually-supplied Pro-Football-Reference exports in
    ``config.PRO_BOWL_DIR``. Selections are keyed primarily on the
    ``Player-additional`` column (the PFR player id) crosswalked to gsis ids via
    nflverse's id map -- exact, no guessing.

    That crosswalk is fantasy-oriented and carries almost no offensive linemen,
    which silently dropped ~19% of selections (nearly all G/T/C/LS). Those fall
    back to a *disambiguated* name match against ``import_players``: the
    normalized name must be unique after filtering to the same position family
    and to players whose career spans the season. Ambiguous names (three
    different "Joe Thomas") are left unmatched rather than guessed.

    Seasons the CSVs don't cover (2020) are filled from the sidecar ``*-gsis.json``.
    """
    import glob
    import json
    import re

    console.step("Importing Pro Bowl selections")
    ids = fetch.ids()[["pfr_id", "gsis_id"]].dropna()
    pfr_to_gsis = dict(zip(ids["pfr_id"], ids["gsis_id"]))

    # Name-based fallback index: (normalized name, position family) -> candidates
    players = fetch.players()[
        ["gsis_id", "display_name", "position", "rookie_season", "last_season"]
    ].dropna(subset=["gsis_id", "display_name"])
    by_name = {}
    for r in players.itertuples(index=False):
        key = (_norm_name(r.display_name), _pos_family(r.position))
        by_name.setdefault(key, []).append(
            (r.gsis_id, r.rookie_season, r.last_season)
        )

    def _fallback(name, pos, season):
        cands = by_name.get((_norm_name(name), _pos_family(pos)), [])
        hits = []
        for gsis, rookie, last in cands:
            lo = rookie if pd.notna(rookie) else -np.inf
            hi = last if pd.notna(last) else np.inf
            if lo <= season <= hi:
                hits.append(gsis)
        return hits[0] if len(set(hits)) == 1 else None

    pro_bowlers, unmatched, recovered = {}, 0, 0
    for path in sorted(glob.glob(str(config.PRO_BOWL_DIR / "pro-bowlers*.csv"))):
        m = re.search(r"(\d{4})", str(path))
        if not m:
            continue
        season = int(m.group(1))
        df = pd.read_csv(path)
        if "Player-additional" not in df.columns:
            continue
        gsis = set()
        for _, row in df.iterrows():
            pfr = str(row.get("Player-additional", "")).strip()
            g = pfr_to_gsis.get(pfr)
            if not g:
                g = _fallback(row.get("Player", ""), row.get("Pos", ""), season)
                if g:
                    recovered += 1
            if g:
                gsis.add(g)
            else:
                unmatched += 1
        pro_bowlers[season] = gsis

    # Fill any season the CSV source is missing (2020).
    for sidecar in sorted(glob.glob(str(config.PRO_BOWL_DIR / "*-gsis.json"))):
        with open(sidecar) as fh:
            extra = json.load(fh)
        for season_str, gsis in extra.items():
            season = int(season_str)
            if season not in pro_bowlers and season in config.PBP_YEARS:
                pro_bowlers[season] = set(gsis)

    seasons = sorted(pro_bowlers)
    if not seasons:
        # The Pro-Football-Reference exports are not distributed with the repo
        # (see pro-bowl/README.md). Without them every pb_* feature is simply
        # zero and the Pro Bowl absence correction is a no-op -- the pipeline
        # still runs end to end, it just loses that one signal.
        console.info(
            f"pro bowlers: no exports found in {config.PRO_BOWL_DIR} -- "
            "pb_* features will be all-zero (see pro-bowl/README.md)"
        )
        return {}
    console.info(
        f"pro bowlers: {len(seasons)} seasons {seasons[0]}-{seasons[-1]}, "
        f"{sum(len(v) for v in pro_bowlers.values()):,} selections "
        f"({recovered} recovered by name+position, {unmatched} still unmatched)"
    )
    return pro_bowlers


def load_schedules():
    """Import game schedules (rest days, neutral site, venue, spread)."""
    console.step("Importing schedules")
    schedules = fetch.schedules(config.PBP_YEARS)
    console.info(f"schedules: {len(schedules):,} rows")
    return schedules


def epa_wpa_columns(play_by_play_df):
    """Discover EPA and WPA columns, dropping the Vegas WPA variants."""
    cols_with_epa = [c for c in play_by_play_df.columns if "epa" in c.lower()]
    cols_with_wpa = [c for c in play_by_play_df.columns if "wpa" in c.lower()]
    remove = ["vegas_wpa", "vegas_home_wpa"]
    cols_with_wpa = [c for c in cols_with_wpa if c not in remove]
    return cols_with_epa, cols_with_wpa
