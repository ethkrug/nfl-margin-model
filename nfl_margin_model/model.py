"""Targets, preprocessing, train/val/test split, and model training."""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from . import config, console


# --------------------------------------------------------------------------
# Targets
# --------------------------------------------------------------------------
def add_targets(team_games_roll, team_games, game_df):
    """Attach scores and the regression/classification targets.

    Returns ``(team_games_roll, team_games)`` since ``team_games`` gains score
    columns used to derive points for/against.
    """
    # Flag each team's opener in the FIRST available season (no prior history, so
    # its rolled features are just the empty seed) to drop before modeling.
    first_season = team_games_roll["season"].min()
    team_games_roll["drop_for_model"] = (
        (team_games_roll["season"] == first_season)
        & (
            team_games_roll.groupby("team")["week"].transform("min")
            == team_games_roll["week"]
        )
    )

    scores = game_df[
        ["game_id", "home_team", "away_team", "home_score", "away_score"]
    ].copy()
    team_games = team_games.merge(scores, on="game_id", how="left")

    team_games_roll["points_for"] = np.where(
        team_games["team"] == team_games["home_team"],
        team_games["home_score"],
        team_games["away_score"],
    )
    team_games_roll["points_against"] = np.where(
        team_games["team"] == team_games["home_team"],
        team_games["away_score"],
        team_games["home_score"],
    )

    team_games_roll["y_margin"] = (
        team_games_roll["points_for"] - team_games_roll["points_against"]
    )
    team_games_roll["y_win"] = (team_games_roll["y_margin"] > 0).astype(int)
    return team_games_roll, team_games


# --------------------------------------------------------------------------
# Model frame
# --------------------------------------------------------------------------
def build_model_frame(team_games_roll):
    """Drop flagged rows and leakage/ID columns, returning the model-ready frame."""
    model_df = team_games_roll[~team_games_roll["drop_for_model"]].copy()
    cols_to_drop = [c for c in config.DROP_BEFORE_MODEL if c in model_df.columns]
    return model_df.drop(columns=cols_to_drop)


# --------------------------------------------------------------------------
# Pro Bowl absence correction
# --------------------------------------------------------------------------
def fit_pb_correction(train_frame, feature_cols, feature=None, oof_start=None):
    """Fit the pooled linear slope for the Pro Bowl absence correction.

    The tree model cannot use the Pro Bowl absence columns -- the situation is
    too rare to isolate among ~169 features -- but it leaves a systematic
    residual: teams missing Pro Bowlers underperform their prediction. This
    estimates that residual's slope explicitly.

    Fitting on the tree's *training* residuals would be biased, since the model
    has already fit those rows. So residuals are built **out of fold**: each
    season from ``oof_start`` is predicted by a model trained only on the
    seasons before it, exactly reproducing the "unseen game" conditions the
    correction will face at prediction time. Those residuals are then pooled
    across every such season and one slope is fit on the lot (~6k rows), which
    is far more stable than calibrating on any single season.

    ``train_frame`` must contain only seasons the caller is allowed to train
    on, so no future information reaches the slope.

    Returns ``{"feature", "beta", "n"}``, or ``None`` when it cannot be fit.
    """
    from sklearn.linear_model import LinearRegression

    feature = feature or config.PB_CORRECTION_FEATURE
    oof_start = oof_start or config.PB_CORRECTION_OOF_START
    if not config.PB_CORRECTION or feature not in train_frame.columns:
        return None

    labeled = train_frame.dropna(subset=["y_margin"])
    seasons = [s for s in sorted(labeled["season"].unique()) if s >= oof_start]
    xs, rs = [], []
    for season in seasons:
        prior = labeled[labeled["season"] < season]
        held = labeled[labeled["season"] == season]
        if prior.empty or held.empty:
            continue
        fold = Pipeline([
            ("prep", build_preprocessor()),
            ("xgb", XGBRegressor(**config.XGB_PARAMS)),
        ]).fit(prior[feature_cols], prior["y_margin"])
        xs.append(held[[feature]].to_numpy(dtype=float))
        rs.append(held["y_margin"].to_numpy(dtype=float) - fold.predict(held[feature_cols]))

    if not xs:
        return None
    X = np.vstack(xs)
    r = np.concatenate(rs)
    # No intercept: with zero differential there is nothing to correct, so the
    # adjustment must be exactly zero rather than a fitted global offset.
    beta = float(
        LinearRegression(fit_intercept=False).fit(X, r).coef_[0]
    )
    console.info(
        f"pb correction: beta={beta:+.3f} on {feature} "
        f"(pooled out-of-fold residuals, n={len(r):,}, seasons "
        f"{seasons[0]}-{seasons[-1]})"
    )
    return {"feature": feature, "beta": beta, "n": int(len(r))}


def apply_pb_correction(predictions, frame, correction, no_flip=None):
    """Add the Pro Bowl absence adjustment to team-perspective predictions.

    Two gates bound what this can do:

    * differentials below ``PB_CORRECTION_MIN_ABS`` are untouched, so a game
      with no Pro Bowl absence imbalance is returned bit-identical;
    * with ``PB_CORRECTION_NO_FLIP`` the adjustment may never carry a
      prediction across zero -- rows where it would are reverted to their
      uncorrected value, so the correction can sharpen a margin but never
      change who is favoured.
    """
    predictions = np.asarray(predictions, dtype=float)
    if not correction:
        return predictions
    feature = correction["feature"]
    if feature not in frame.columns:
        return predictions
    x = frame[feature].to_numpy(dtype=float)
    adjustment = correction["beta"] * x
    gated = np.where(np.abs(x) >= config.PB_CORRECTION_MIN_ABS, adjustment, 0.0)
    adjusted = predictions + gated

    if no_flip is None:
        no_flip = config.PB_CORRECTION_NO_FLIP
    if no_flip:
        flips = (((predictions > 0) & (adjusted <= 0))
                 | ((predictions < 0) & (adjusted >= 0)))
        adjusted = np.where(flips, predictions, adjusted)
    return adjusted


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------
def build_preprocessor():
    """One-hot encode the categorical context columns; pass the rest through."""
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[("cat", cat_pipe, config.CATEGORICAL_COLS)],
        remainder="passthrough",
    )


# --------------------------------------------------------------------------
# Split
# --------------------------------------------------------------------------
def split_data(final):
    """Split by season into train / validation / test feature & target sets."""
    train = final[final["season"] <= config.TRAIN_MAX]
    val = final[final["season"] == config.VAL_SEASON]
    test = final[final["season"] == config.TEST_SEASON]

    splits = {}
    for name, frame in (("train", train), ("val", val), ("test", test)):
        X = frame.drop(columns=["y_margin", "y_win"])
        y = frame["y_margin"]
        splits[name] = (X, y)
    return splits


# --------------------------------------------------------------------------
# Train + evaluate
# --------------------------------------------------------------------------
def _evaluate(label, model, X, y):
    pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, pred))
    mae = mean_absolute_error(y, pred)
    r2 = r2_score(y, pred)
    console.metrics_table(label, rmse, mae, r2)
    return {"rmse": rmse, "mae": mae, "r2": r2}


def train_and_evaluate(splits, preprocessor):
    """Fit the tuned XGBoost pipeline and report validation/test metrics."""
    best_model = Pipeline([
        ("prep", preprocessor),
        ("xgb", XGBRegressor(**config.XGB_PARAMS)),
    ])

    X_tr, y_tr = splits["train"]
    best_model.fit(X_tr, y_tr)

    metrics = {
        "val": _evaluate(f"{config.VAL_SEASON} Validation", best_model, *splits["val"]),
        "test": _evaluate(f"{config.TEST_SEASON} Test", best_model, *splits["test"]),
    }
    return best_model, metrics
