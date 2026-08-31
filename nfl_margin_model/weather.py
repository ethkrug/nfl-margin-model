"""Weather for projected (not-yet-played) games.

Actual game weather is only recorded at/after kickoff, so upcoming games need an
estimate. Priority per game: indoor -> neutral 70/0; a live game-day forecast
when kickoff is within ``FORECAST_HORIZON_DAYS``; else the month-specific stadium
climatology; else the all-season stadium mean.
"""
from __future__ import annotations

from datetime import date

import pandas as pd


# Home stadium coordinates (approx), for the game-day forecast lookup.
STADIUM_LATLON = {
    "ARI": (33.528, -112.263), "ATL": (33.755, -84.401), "BAL": (39.278, -76.623),
    "BUF": (42.774, -78.787), "CAR": (35.226, -80.853), "CHI": (41.862, -87.617),
    "CIN": (39.095, -84.516), "CLE": (41.506, -81.700), "DAL": (32.747, -97.094),
    "DEN": (39.744, -105.020), "DET": (42.340, -83.046), "GB": (44.501, -88.062),
    "HOU": (29.685, -95.411), "IND": (39.760, -86.164), "JAX": (30.324, -81.637),
    "KC": (39.049, -94.484), "LV": (36.091, -115.183), "LAC": (33.954, -118.339),
    "LA": (33.954, -118.339), "MIA": (25.958, -80.239), "MIN": (44.974, -93.258),
    "NE": (42.091, -71.264), "NO": (29.951, -90.081), "NYG": (40.814, -74.074),
    "NYJ": (40.814, -74.074), "PHI": (39.901, -75.168), "PIT": (40.447, -80.016),
    "SEA": (47.595, -122.332), "SF": (37.403, -121.970), "TB": (27.976, -82.503),
    "TEN": (36.166, -86.771), "WAS": (38.908, -76.865),
}
FORECAST_HORIZON_DAYS = 7   # use a live forecast only within this many days


def month_climatology(gdf, sched, precip_played):
    """(home_team, month) -> (temp, wind, precip_rate) from completed games.

    Month-specific, so a September opener uses September norms rather than a
    year-round average that would drag in December cold. Uses game_df weather
    (pbp-sourced, well populated) with the kickoff month from the schedule.
    """
    dates = sched[["game_id", "gameday"]].dropna(subset=["gameday"])
    d = gdf[["game_id", "home_team", "temp", "wind"]].merge(dates, on="game_id", how="inner")
    d["month"] = d["gameday"].astype(str).str.slice(5, 7)
    d = d.merge(precip_played, on="game_id", how="left")
    g = (d.groupby(["home_team", "month"])
         .agg(temp=("temp", "mean"), wind=("wind", "mean"), precip=("precip", "mean"))
         .reset_index())
    return {(r.home_team, r.month): (r.temp, r.wind, r.precip) for r in g.itertuples()}


def forecast_openmeteo(lat, lon, datestr):
    """Open-Meteo daily forecast (free, no key) -> (temp_F, wind_mph, precip 0/1)."""
    import json
    import ssl
    import urllib.request
    url = ("https://api.open-meteo.com/v1/forecast"
           f"?latitude={lat}&longitude={lon}"
           "&daily=temperature_2m_mean,wind_speed_10m_max,precipitation_sum"
           "&temperature_unit=fahrenheit&wind_speed_unit=mph&timezone=America%2FNew_York"
           f"&start_date={datestr}&end_date={datestr}")
    ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
    d = json.loads(urllib.request.urlopen(url, timeout=15, context=ctx).read())["daily"]
    return (float(d["temperature_2m_mean"][0]), float(d["wind_speed_10m_max"][0]),
            1 if (d["precipitation_sum"][0] or 0) > 0.1 else 0)


def impute_proj_weather(ctx, gdf, clim, ref_date, use_forecast=True):
    """Fill future-game temp/wind/precip on the projected context rows."""
    ctx = ctx.copy()
    t_all = gdf.groupby("home_team")["temp"].mean()
    w_all = gdf.groupby("home_team")["wind"].mean()
    tg, wg, pg, src = [], [], [], []
    for r in ctx.itertuples():
        team, roof, gd = r.home_team, r.roof, str(r.gameday)
        month = gd[5:7] if len(gd) >= 7 else None
        if roof in ("dome", "closed"):
            tg.append(70.0); wg.append(0.0); pg.append(0); src.append("indoor"); continue
        got = None
        if use_forecast and team in STADIUM_LATLON and len(gd) >= 10:
            try:
                days = (date.fromisoformat(gd[:10]) - ref_date).days
                if 0 <= days <= FORECAST_HORIZON_DAYS:
                    lat, lon = STADIUM_LATLON[team]
                    got = forecast_openmeteo(lat, lon, gd[:10]); src.append("forecast")
            except Exception:
                got = None
        if got is None:
            c = clim.get((team, month))
            if c and pd.notna(c[0]):
                got = (c[0], c[1], 0 if pd.isna(c[2]) else round(float(c[2]), 2))
                src.append("climatology")
            else:
                got = (t_all.get(team, gdf["temp"].mean()), w_all.get(team, gdf["wind"].mean()), 0)
                src.append("season-mean")
        tg.append(got[0]); wg.append(got[1]); pg.append(got[2])
    ctx["temp"], ctx["wind"], ctx["precip"] = tg, wg, pg
    ctx["_wsrc"] = src
    return ctx
