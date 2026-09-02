"""Resilient nflverse downloads.

Every feed the model uses is a parquet/CSV pulled over HTTPS on demand, and
those fetches fail transiently: GitHub's edge drops a connection mid-handshake
(``ssl.SSLEOFError: UNEXPECTED_EOF_WHILE_READING``), a release redirect times
out, a load balancer answers 502. ``nfl_data_py`` handles that badly, in two
different ways:

* ``import_injuries``/``import_depth_charts``/``import_schedules`` let the error
  propagate, so one dropped connection aborts a build minutes in.
* ``import_pbp_data`` *swallows* a per-season failure -- it prints ``Data not
  available for 2011`` and returns a frame with that season simply absent. The
  pipeline then trains on a hole in its history and reports success, which is
  far worse than crashing.

So every fetch goes through :func:`call` (retry transient failures with backoff)
and every year-keyed feed through :func:`by_year`, which insists that each
requested season actually arrived. A season may only be missing when the caller
declares it ``optional`` -- the upcoming season, whose files are not published
until it exists.
"""

from __future__ import annotations

import http.client
import socket
import ssl
import time
import urllib.error

import pandas as pd
import nfl_data_py as nfl

from . import console

ATTEMPTS = 4          # 1 try + 3 retries
BACKOFF = 2.0         # seconds before the first retry, doubling each time


class FetchError(RuntimeError):
    """A feed could not be retrieved (after retries) or came back empty."""


def _is_missing(exc):
    """True when the server said the file is not there (vs. a flaky connection).

    A 404 is a fact about the release, not a hiccup, so it must not be retried:
    the upcoming season's depth charts simply do not exist yet.
    """
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code == 404
    return isinstance(exc, FileNotFoundError)


def _is_transient(exc):
    """True for network-level failures that a retry has a real chance of fixing."""
    if _is_missing(exc):
        return False
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in (408, 425, 429) or exc.code >= 500
    # SSLError/URLError/socket.timeout are all OSError subclasses; HTTPException
    # covers the "server closed the connection early" family.
    return isinstance(exc, (urllib.error.URLError, ssl.SSLError, socket.timeout,
                            http.client.HTTPException, OSError, FetchError))


def call(what, fn, *args, **kwargs):
    """Run ``fn(*args, **kwargs)``, retrying transient network failures.

    ``what`` is a human label for the log line ("injuries 2011"). Anything that
    is not a transient network failure -- a schema error, a 404 -- is raised
    immediately; retrying those only wastes minutes and hides the real cause.
    """
    delay = BACKOFF
    for attempt in range(1, ATTEMPTS + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if not _is_transient(exc) or attempt == ATTEMPTS:
                if _is_transient(exc):
                    raise FetchError(
                        f"{what}: giving up after {ATTEMPTS} attempts ({exc})"
                    ) from exc
                raise
            console.info(f"{what}: {type(exc).__name__}: {exc} -- "
                         f"retrying in {delay:.0f}s ({attempt}/{ATTEMPTS - 1})")
            time.sleep(delay)
            delay *= 2


def by_year(what, fn, years, optional=()):
    """Fetch ``fn(year)`` for each year, retrying per year, and verify coverage.

    Fetching a year at a time means one flaky season is retried on its own
    rather than re-downloading a decade, and -- the point of this module -- a
    season that never arrives raises instead of quietly shrinking the frame.
    Years listed in ``optional`` may be absent (404) and are skipped with a note.
    """
    optional = set(optional)
    frames, missing = [], []
    for year in years:
        try:
            df = call(f"{what} {year}", fn, year)
        except Exception as exc:
            if _is_missing(exc) and year in optional:
                console.info(f"{what}: {year} not published yet -- skipping")
                continue
            raise FetchError(f"{what}: could not load {year} ({exc})") from exc
        if df is None or not len(df):
            if year in optional:
                console.info(f"{what}: {year} is empty -- skipping")
                continue
            missing.append(year)
            continue
        frames.append(df)
    if missing:
        raise FetchError(f"{what}: no rows returned for {missing}; refusing to "
                         "build on an incomplete history -- rerun when the "
                         "nflverse release is reachable")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# --- the feeds ----------------------------------------------------------
# Thin wrappers so call sites read like the nfl_data_py ones they replace.

def _pbp_year(year, downcast=False, columns=None):
    """One season of play-by-play, converting nfl_data_py's silence into an error.

    ``import_pbp_data`` catches its own exceptions and returns an empty frame,
    so emptiness -- not an exception -- is the failure signal to retry on.
    """
    df = nfl.import_pbp_data([year], columns=columns, downcast=downcast, cache=False)
    if not len(df):
        raise FetchError(f"play-by-play {year}: nfl_data_py returned no rows")
    return df


def pbp(years, downcast=False, columns=None):
    """Play-by-play for every season in ``years`` (all of them, or an error)."""
    return by_year("play-by-play", lambda y: _pbp_year(y, downcast, columns), years)


def injuries(years, optional=()):
    """Weekly injury reports."""
    return by_year("injuries", lambda y: nfl.import_injuries([y]), years, optional)


def depth_charts(years, optional=()):
    """Team depth charts (raw; the schema unification lives in :mod:`data`)."""
    return by_year("depth charts", lambda y: nfl.import_depth_charts([y]),
                   years, optional)


def weekly(years, columns=None, downcast=False):
    """Weekly player stats."""
    return by_year("weekly player data",
                   lambda y: nfl.import_weekly_data([y], columns=columns,
                                                    downcast=downcast), years)


def schedules(years):
    """Schedules for ``years``.

    One CSV covers every season, so there is nothing to verify per year -- and
    nothing to assert either: the upcoming season legitimately has no rows until
    the schedule is released.
    """
    years = list(years)
    return call("schedules", nfl.import_schedules, years)


def team_desc():
    """Team names/colours/ids."""
    return call("team descriptions", nfl.import_team_desc)


def ids():
    """Cross-site player id mapping."""
    return call("player ids", nfl.import_ids)


def players():
    """Descriptive player data."""
    return call("players", nfl.import_players)
