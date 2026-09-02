"""Walk-forward evaluation of the point-margin model.

Reproduces the headline metrics (MSE / RMSE / MOE=MAE / straight-up winners)
using temporal, walk-forward holdouts: for each test season the model is trained
only on prior seasons, so nothing from the future leaks in.

    default splits:  train 2010-2023 -> test 2024
                     train 2010-2024 -> test 2025
(train-start 2010 matches the production window; the earliest season is used only
to seed the next season's rolling features. Metrics are at the team-game level,
the model's native granularity.)

Usage
-----
    # live nflverse data (slower; a minute or two to pull play-by-play):
    python -m nfl_margin_model.evaluate

    # fast, from a directory of cached parquet (pbp/depth/injuries/schedules):
    python -m nfl_margin_model.evaluate --cache /path/to/parquet

    # download once, keeping the raw feeds for later --cache runs:
    python -m nfl_margin_model.evaluate --save-cache /path/to/parquet
"""
from __future__ import annotations

import argparse

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from . import config, model, predict


def _feature_cols(frame):
    drop = [c for c in config.DROP_BEFORE_MODEL if c in frame.columns]
    return [c for c in frame.columns
            if c not in drop + ["y_margin", "y_win", "drop_for_model", "pred"]]


def evaluate(cache=None, train_start=2010, holdouts=(2024, 2025), save_cache=None):
    """Return a dict of per-holdout and pooled metrics (team-game level)."""
    pbp, depth, inj, sched = predict.load_raw(cache, project_season=max(holdouts) + 1,
                                              save_cache=save_cache)
    tgr, _, _ = predict.build_frame(pbp, depth, inj, sched,
                                    project_season=None, train_through=max(holdouts))
    tgr = tgr[tgr["season"] <= max(holdouts)].copy()
    feats = _feature_cols(tgr)

    results = {}
    all_y, all_p = [], []
    for test_season in holdouts:
        train_seasons = list(range(train_start, test_season))
        tr = tgr[tgr["season"].isin(train_seasons)].dropna(subset=["y_margin"])
        te = tgr[tgr["season"] == test_season].dropna(subset=["y_margin"]).copy()
        clf = Pipeline([("prep", model.build_preprocessor()),
                        ("xgb", XGBRegressor(**config.XGB_PARAMS))])
        clf.fit(tr[feats], tr["y_margin"])
        pb_corr = model.fit_pb_correction(tr, feats)
        p = model.apply_pb_correction(clf.predict(te[feats]), te, pb_corr)
        y = te["y_margin"].to_numpy()
        all_y.append(y); all_p.append(p)
        results[test_season] = _metrics(y, p, train_seasons)

    y = np.concatenate(all_y); p = np.concatenate(all_p)
    results["pooled"] = _metrics(y, p, None)
    results["_meta"] = {"n_features": len(feats),
                        "learning_rate": config.XGB_PARAMS["learning_rate"],
                        "max_depth": config.XGB_PARAMS["max_depth"]}
    return results


def _metrics(y, p, train_seasons):
    mse = float(mean_squared_error(y, p))
    return {
        "n": int(len(y)),
        "train": train_seasons,
        "mse": round(mse, 2),
        "rmse": round(mse ** 0.5, 2),
        "moe_mae": round(float(mean_absolute_error(y, p)), 2),
        "winners": round(float(((p > 0) == (y > 0)).mean()), 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None, help="dir of cached parquet (fast)")
    ap.add_argument("--save-cache", default=None, metavar="DIR",
                    help="write the raw nflverse feeds to DIR after downloading "
                         "them, so a later --cache DIR run needs no network")
    ap.add_argument("--train-start", type=int, default=2010)
    ap.add_argument("--holdouts", type=int, nargs="+", default=[2024, 2025])
    args = ap.parse_args()

    res = evaluate(cache=args.cache, train_start=args.train_start,
                   holdouts=tuple(args.holdouts), save_cache=args.save_cache)
    m = res["_meta"]
    print(f"\nMODEL: {m['n_features']} features | lr={m['learning_rate']} depth={m['max_depth']}\n")
    hdr = f"{'holdout':>22} | {'n':>4} | {'MSE':>7} | {'RMSE':>5} | {'MOE(MAE)':>8} | {'winners':>7}"
    print(hdr); print("-" * len(hdr))
    for key in list(args.holdouts) + ["pooled"]:
        r = res[key]
        label = f"{key} (train {r['train'][0]}-{r['train'][-1]})" if r["train"] else "POOLED"
        print(f"{label:>22} | {r['n']:>4} | {r['mse']:>7.2f} | {r['rmse']:>5.2f} | "
              f"{r['moe_mae']:>6.2f} pts | {r['winners']:>6.1%}")


if __name__ == "__main__":
    main()
