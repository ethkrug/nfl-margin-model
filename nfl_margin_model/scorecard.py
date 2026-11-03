"""Week-by-week scorecard: model line vs market line, both graded on the result.

:mod:`evaluate` answers "how good is the model?" with season-sized walk-forward
holdouts. This answers the in-season question instead: on the games the page
shows (the seasons after ``TRAIN_THROUGH``), how far was the model's margin from
the final margin each week, and how far was the market's closing spread? Both
lines are graded against the same actual margin, home perspective, so the
columns are directly comparable.

Only finished games that carry a market spread are scored; a final without a
line is counted separately rather than silently dropped. Errors are per game,
which for MAE/RMSE equals :mod:`evaluate`'s team-game level (the two rows of a
game carry mirror-image errors).

Usage
-----
    # rerun the model on live data (same as a frontend refresh):
    python -m nfl_margin_model.scorecard

    # fast, from cached parquet:
    python -m nfl_margin_model.scorecard --cache /path/to/parquet

    # no retraining: score a payload generate.py already wrote:
    python -m nfl_margin_model.scorecard --preds nfl_margin_model/frontend/preds.json

    # also write the table out:
    python -m nfl_margin_model.scorecard --csv scorecard.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

# Allow running as a plain script (add project root so the package imports).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nfl_margin_model import predict  # noqa: E402


def _errors(records, seasons=None):
    """Finished games with a market line, one row each, with both errors."""
    df = pd.DataFrame(records)
    if seasons:
        df = df[df["season"].isin(seasons)]
    done = df[df["actual_margin"].notna()].copy()
    no_line = done["market_spread"].isna()
    scored = done[~no_line].copy()
    scored["model_err"] = scored["pred_margin"] - scored["actual_margin"]
    scored["market_err"] = scored["market_spread"] - scored["actual_margin"]
    return scored, int(no_line.sum())


def _summary(g):
    """MAE/RMSE/winners for one group of games, model and market side by side."""
    y = g["actual_margin"].to_numpy(float)
    me, ke = g["model_err"].to_numpy(float), g["market_err"].to_numpy(float)

    def winners(line):
        # A tie or a pick'em line is a call nobody made; leave it out.
        line = g[line].to_numpy(float)
        ok = (line != 0) & (y != 0)
        return float(((line[ok] > 0) == (y[ok] > 0)).mean()) if ok.any() else np.nan

    model_mae, market_mae = np.abs(me).mean(), np.abs(ke).mean()
    model_rmse, market_rmse = np.sqrt((me ** 2).mean()), np.sqrt((ke ** 2).mean())
    return {
        "n": len(g),
        "model_mae": model_mae, "market_mae": market_mae,
        "mae_diff": model_mae - market_mae,
        "model_rmse": model_rmse, "market_rmse": market_rmse,
        "rmse_diff": model_rmse - market_rmse,
        "model_winners": winners("pred_margin"),
        "market_winners": winners("market_spread"),
        # How often the model's line landed closer to the result than the market's.
        "model_closer": float((np.abs(me) < np.abs(ke)).mean()),
    }


def scorecard(records, seasons=None):
    """Return ``(table, n_no_line)``: per-week rows, then one season-to-date row
    per season. ``*_diff`` is model minus market, so negative means the model
    was the more accurate line."""
    scored, n_no_line = _errors(records, seasons)
    if scored.empty:
        return pd.DataFrame(), n_no_line
    rows = []
    for season, sg in scored.groupby("season"):
        for week, wg in sg.groupby("week"):
            rows.append({"season": season, "week": str(week), **_summary(wg)})
        rows.append({"season": season, "week": "total", **_summary(sg)})
    table = pd.DataFrame(rows)
    table["n"] = table["n"].astype(int)
    return table, n_no_line


def _load_records(args):
    if args.preds:
        with open(args.preds, encoding="utf-8") as f:
            payload = json.load(f)
        return payload["games"], payload.get("meta", {})
    return predict.predict_games(cache=args.cache, train_through=args.train_through,
                                 save_cache=args.save_cache)


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--cache", default=None, help="dir of cached parquet (fast)")
    src.add_argument("--preds", default=None, metavar="JSON",
                     help="score an existing preds.json / preds_no_logos.json "
                          "instead of rerunning the model")
    ap.add_argument("--save-cache", default=None, metavar="DIR",
                    help="write the raw nflverse feeds to DIR after downloading them")
    ap.add_argument("--train-through", type=int, default=predict.TRAIN_THROUGH)
    ap.add_argument("--season", type=int, nargs="+", default=None,
                    help="limit to these seasons (default: every season shown)")
    ap.add_argument("--csv", default=None, metavar="PATH", help="also write the table")
    args = ap.parse_args()

    records, meta = _load_records(args)
    table, n_no_line = scorecard(records, seasons=args.season)
    if meta.get("generated"):
        print(f"\npredictions generated {meta['generated']}, "
              f"trained through {meta.get('train_through')}")
    if table.empty:
        print("no finished games with a market line to score yet")
        return

    hdr = (f"{'season':>6} {'week':>5} | {'n':>3} | {'MAE mdl':>7} {'mkt':>5} {'diff':>6} | "
           f"{'RMSE mdl':>8} {'mkt':>5} {'diff':>6} | {'SU mdl':>6} {'mkt':>5} | {'closer':>6}")
    print(); print(hdr); print("-" * len(hdr))
    for r in table.itertuples():
        if r.week == "total":
            print("-" * len(hdr))
        print(f"{r.season:>6} {r.week:>5} | {r.n:>3} | "
              f"{r.model_mae:>7.2f} {r.market_mae:>5.2f} {r.mae_diff:>+6.2f} | "
              f"{r.model_rmse:>8.2f} {r.market_rmse:>5.2f} {r.rmse_diff:>+6.2f} | "
              f"{r.model_winners:>6.1%} {r.market_winners:>5.1%} | {r.model_closer:>6.1%}")
    print("\ndiff = model - market (negative: model more accurate). SU = straight-up "
          "winners. closer = share of games the model's line beat the market's.")
    if n_no_line:
        print(f"{n_no_line} finished game(s) had no market line and are not scored.")

    if args.csv:
        table.to_csv(args.csv, index=False, float_format="%.4f")
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
