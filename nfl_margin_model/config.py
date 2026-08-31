"""Central configuration for the NFL point-margin model pipeline.

Every knob that the original notebook scattered across cells lives here so the
pipeline can be tuned in one place. Changing these values changes the model;
the defaults reproduce the notebook exactly.
"""

from pathlib import Path

# --- Pro Bowl pedigree ---------------------------------------------------
# Manually-supplied Pro-Football-Reference Pro Bowl exports, one CSV per season
# (``pro-bowlers<SEASON>.csv``; filename year = NFL season). 2020 is absent from
# that source and is filled from ``pro-bowlers2020-gsis.json``.
PRO_BOWL_DIR = Path(__file__).resolve().parent / "pro-bowl"
# "Recently a Pro Bowler" = selected in any of the last N seasons.
PB_RECENT_YEARS = 3
# "A Pro Bowler X times in Y years" -- the sustained-quality definition.
PB_ELITE_COUNT = 3
PB_ELITE_WINDOW = 8

# --- Data windows --------------------------------------------------------
# Weekly player data and play-by-play data are pulled over these seasons.
WEEKLY_YEARS = range(2020, 2025)
# Training history starts 2010: walk-forward tests showed extending back to ~2010
# lowered holdout RMSE on both 2024 and 2025 (plateaus there; 2005-2009 add no
# accuracy and lack injury/pressure coverage). Injuries exist from 2009, so a
# 2010 start keeps all feeds populated.
PBP_YEARS = range(2010, 2026)

# --- Rolling team features ----------------------------------------------
# Windows (in games) used to roll opponent-adjusted team stats.
ROLL_WINDOWS = (5, 10)

# Prior->current blend for the early-season rolling window. Before a team has a
# full window of current-season games, the empty slots are filled by the
# previous-season seed, each empty slot weighted ``w_prior`` relative to one
# current-season game:
#   value = (w_prior*empty*seed + sum(current in window)) / (w_prior*empty + count)
# Higher -> last season lingers longer; once the window fills with current games
# the prior drops out entirely, so this only touches the first few weeks.
#
# The best weight differs by phase (2023-25 backtests): weeks 2-4 (a 1-3 game
# current sample, very noisy) want a strong prior ~3-5; from week 5 on the
# current sample is reliable and a light prior ~1 is best. So the weight is a
# function of the game's week rather than one global value.
ROLL_PRIOR_WEIGHT_EARLY = 4.0     # weeks <= ROLL_PRIOR_EARLY_MAX_WEEK
ROLL_PRIOR_WEIGHT_LATE = 1.0      # later weeks
ROLL_PRIOR_EARLY_MAX_WEEK = 4


def roll_prior_weight(week):
    """Per-game prior weight: strong prior early, light prior once settled.

    (Week 1 is the pure prior seed regardless of the weight, so its value here
    only matters via how prior seasons' early weeks are modelled in training.)
    """
    return ROLL_PRIOR_WEIGHT_EARLY if week <= ROLL_PRIOR_EARLY_MAX_WEEK else ROLL_PRIOR_WEIGHT_LATE

# --- Starting-QB features ------------------------------------------------
QB_WINDOWS = (5, 10)            # rolling windows over a QB's own starts
TRAIN_MAX_SEASON = 2023         # replacement baseline uses train-era games only
REPLACEMENT_QUANTILE = 0.15     # replacement-level = this pct of train-era starter EPA
QB_SHRINKAGE_K = 3.0            # shrinkage strength (in games) toward replacement level
# Career-anchored shrinkage: also pull a QB's rolling form toward their OWN
# career-to-date mean (not just league replacement), so an elite QB in a slump
# (e.g. Lamar) isn't rated near replacement. Weight in games; 0.0 = off (shrink
# only toward replacement, the original behavior). Reformulates qb_quality in
# place -- adds no features.
QB_CAREER_SHRINK = 0.0
# Deviation-triggered career regression (surgical alternative): only pull toward
# career when a QB with an ESTABLISHED career (>= QB_DEV_MIN_STARTS) has recent
# form deviating from it by more than QB_DEV_THRESHOLD; then move a fraction
# QB_DEV_REG_LAMBDA of the gap. 0.0 = off. Leaves ordinary QBs untouched.
QB_DEV_REG_LAMBDA = 0.0
QB_DEV_MIN_STARTS = 25
QB_DEV_THRESHOLD = 0.10
# Longer-memory blend (option 3): mix the short rolling form with a longer
# QB_LONG_WINDOW-game window before shrinkage, by weight QB_LONG_BLEND (0 = off).
# A middle ground between recent form and all-time career mean.
QB_LONG_WINDOW = 25
QB_LONG_BLEND = 0.0

# QB offseason regression: a QB's rolling form carries across the offseason
# without resetting, so a QB who ended the year slumping/injured enters the new
# year underrated. At a QB's season opener (0 current-season starts) their quality
# is pulled toward their career-to-date baseline by ``QB_OFFSEASON_REG_STRENGTH``,
# fading linearly to 0 by ``QB_OFFSEASON_REG_FADE`` current-season starts:
#   r(s) = STRENGTH * max(0, 1 - s / FADE);  quality = (1-r)*form + r*career
# ENABLED at STRENGTH=0.75, FADE=1.0. FADE is the auto-shutoff and is the whole
# reason this works: at 1.0 the pull applies to a QB's season OPENER only and is
# exactly 0 from his 2nd start on, so it cannot leak into midseason. An earlier
# backtest rejected this mechanism, but it only ever tried FADE=5.0, which kept
# regressing through five starts and is what damaged weeks 5-13.
#
# Retrained walk-forward sweep (2021-25, STRENGTH x FADE, RMSE by phase):
#   FADE=1: wk1 -0.032/-0.074/-0.157 at S=0.25/0.50/0.75, overall +0.002/+0.001/-0.008
#   FADE=2,3,5: every combination degrades overall RMSE (+0.006 .. +0.019)
# So FADE=1 is free -- weeks 5-13 improve slightly (12.722 -> 12.702) and overall
# RMSE is flat-to-better. STRENGTH is set where the week-1 gain plateaus: 0.90 is
# identical to 0.75, and 1.00 buys 0.006 more on week 1 while costing overall
# RMSE, so 0.75 is the knee and it keeps 25% weight on recent form.
#
# Honest caveat: pooled week 1 is only 160 team-games (SE ~0.7), so -0.157 is well
# inside noise on its own. What justifies it is the clean dose-response, the
# absence of any cost elsewhere, and the mechanism: a season opener has NO
# current-season data, so career talent is genuinely the better estimate. Per
# season the week-1 gain is 4-of-5 (2021 -0.32, 2023 -0.43, 2025 -0.60, 2022
# +0.09) with 2024 the exception (+0.54, and worsening as STRENGTH rises).
QB_OFFSEASON_REG_STRENGTH = 0.75
QB_OFFSEASON_REG_FADE = 1.0

# A designated starter (depth-chart QB1) is considered unavailable for a game
# when their injury report status is one of these (forces a backup to start).
QB_INACTIVE_STATUSES = ("Out", "Doubtful")
# Depth rank assigned to a passer we can't locate on that week's QB depth chart.
QB_DEPTH_SENTINEL = 3

# --- Train / validation / test split (by season) ------------------------
TRAIN_MAX = 2023                # train: season <= TRAIN_MAX
VAL_SEASON = 2024               # validation: season == VAL_SEASON
TEST_SEASON = 2025              # test: season == TEST_SEASON

# --- Game-level context columns kept from the play-by-play ---------------
GAME_FEATURES = [
    "season", "season_type", "week", "div_game", "roof", "surface",
    "temp", "wind", "away_score", "home_score",
]

# --- Per-team, per-game counting stats summed from the play-by-play ------
# (Duplicates are intentional in the source; they are de-duplicated on use.)
SUM_COLS = [
    "punt_blocked",
    "third_down_converted",
    "third_down_failed",
    "fourth_down_converted",
    "fourth_down_failed",
    "interception",
    "touchback",
    "punt_inside_twenty",
    "punt_in_endzone",
    "kickoff_inside_twenty",
    "kickoff_in_endzone",
    "fumble_forced",
    "safety",
    "penalty",
    "tackled_for_loss",
    "fumble_lost",
    "qb_hit",
    "own_kickoff_recovery",
    "qb_hit",
    "rush_attempt",
    "pass_attempt",
    "sack",
    "touchdown",
    "pass_touchdown",
    "rush_touchdown",
    "return_touchdown",
    "extra_point_attempt",
    "two_point_attempt",
    "field_goal_attempt",
    "kickoff_attempt",
    "punt_attempt",
    "fumble",
    "complete_pass",
]

# Engineered/leakage columns dropped before the model sees the frame.
DROP_BEFORE_MODEL = [
    "game_id", "team", "opponent", "for_epa", "for_rush_epa",
    "for_pass_epa", "for_comp_air_epa", "for_comp_yac_epa",
    "for_raw_air_epa", "for_raw_yac_epa", "for_rush_wpa", "for_pass_wpa",
    "for_comp_air_wpa", "for_comp_yac_wpa", "for_raw_air_wpa",
    "for_raw_yac_wpa", "allow_epa", "allow_rush_epa", "allow_pass_epa",
    "allow_comp_air_epa", "allow_comp_yac_epa", "allow_raw_air_epa",
    "allow_raw_yac_epa", "allow_rush_wpa", "allow_pass_wpa",
    "allow_comp_air_wpa", "allow_comp_yac_wpa", "allow_raw_air_wpa",
    "allow_raw_yac_wpa", "points_for", "points_against", "drop_for_model",
    # QB cols not fed directly to the model (ids/names, current-game
    # dropbacks, and raw rolls superseded by the shrunk qb_quality_*).
    # The model-facing QB features that survive are qb_quality_5/10,
    # qb_depth_order, is_designated_starter, starter_inactive, qb_prior_starts.
    "qb_player_id", "qb_player_name", "qb_dropbacks",
    "qb_roll_epa_5", "qb_roll_epa_10", "qb_career_epa",
    # Both of these are computed from the QB who ACTUALLY took the most
    # dropbacks, which is only knowable after kickoff, so neither can be a
    # model input. They are kept on the frame as diagnostics (they drive the
    # "designated QB1 started X%" line). The pre-game half of the same signal --
    # starter_inactive, from the injury report on the named QB1 -- is a feature.
    "qb_depth_order", "is_designated_starter",
    # Pre-offseason-pull QB quality: projection-path scaffolding, not a feature
    # (it is a near-duplicate of qb_quality_* and would just add noise).
    "qb_quality_base_5", "qb_quality_base_10",
    # Pro Bowl pedigree columns. As *tree* inputs these are null-to-harmful
    # (walk-forward: +0.003 mean RMSE) -- the situations are far too rare for a
    # depth-2 ensemble to carve out among 167 other features. They are kept on
    # the frame (not fed to the model) because the effect is real and is
    # exploited instead by a pooled linear correction on edge_pb_out_ever.
    "pb_out_ever", "pb_out_recent", "pb_out_elite",
    "pb_out_ever_nonqb", "pb_out_recent_nonqb", "pb_out_elite_nonqb",
    "pb_out_ol",
    "pb_lost_ever", "pb_gain_ever", "pb_net_ever",
    "pb_lost_recent", "pb_gain_recent", "pb_net_recent",
    "pb_lost_elite", "pb_gain_elite", "pb_net_elite",
    "edge_pb_out_ever", "edge_pb_out_recent", "edge_pb_out_elite",
    "edge_pb_out_ever_nonqb", "edge_pb_out_recent_nonqb",
    "edge_pb_out_elite_nonqb", "edge_pb_out_ol",
    "edge_pb_net_ever", "edge_pb_net_recent", "edge_pb_net_elite",
]

# --- Pro Bowl absence correction ----------------------------------------
# A team missing Pro Bowl-pedigree players underperforms the tree model by a
# real margin (~2.4 pts when 2+ are out), but the situation is far too rare
# (~6% of team-games) for a depth-2 ensemble to isolate among 167 features --
# fed in as tree inputs the columns are null-to-harmful. Instead the residual
# is removed with an explicit linear term whose slope is pooled across many
# out-of-fold seasons. Walk-forward: helps 5/5 test seasons, and scales with
# the differential (>=2 out: -0.19 RMSE / -0.13 MAE / +1.1pp winners).
PB_CORRECTION = True
# The differential this corrects: opponent's ruled-out Pro Bowlers minus mine.
# Antisymmetric by construction, so correcting each team-perspective row and
# antisymmetrically averaging yields exactly beta * edge for the game.
PB_CORRECTION_FEATURE = "edge_pb_out_ever"
# Only adjust when the differential is at least this large. The counts are
# integers so this is equivalent to "non-zero", and with no intercept the
# adjustment is already exactly 0 there -- the gate makes that a guarantee
# rather than an accident, so games without a PB absence are never touched.
PB_CORRECTION_MIN_ABS = 1.0
# First season to hold out when building out-of-fold residuals. Each season
# from here on is predicted by a model trained only on seasons before it; the
# slope is fit on all of those residuals pooled (~6k rows). Single-season
# calibration is far too noisy -- beta swung 0.38-2.56 season to season.
PB_CORRECTION_OOF_START = 2014
# Never let the adjustment change who is favoured. The correction improves
# margin accuracy (RMSE/MAE on 5/5 seasons) but nudging a near-zero margin
# across zero was costing straight-up winner accuracy (2025: -1.9pp), and the
# app leads with picks and moneylines. When the adjustment would flip the sign
# the row keeps its uncorrected prediction, so picks are guaranteed identical
# to the uncorrected model while every non-flipping game still gets corrected.
PB_CORRECTION_NO_FLIP = True

# Categorical columns one-hot encoded by the preprocessor.
CATEGORICAL_COLS = ["surface", "roof", "season_type"]

# Keep the opponent's raw rolled stats (opp_* except opp_qb_*) as features? They
# are redundant with the edge_* differentials built from them and add overfitting
# noise, so they are dropped by default (backtests show a small RMSE improvement).
# Set True to restore them.
KEEP_OPP_RAW = False

# Keep the own-team rolled EPA/WPA columns (for_/allow *_epa/_wpa *_roll*)? They
# are redundant with the edge_* EPA differentials built from them. Dropping them
# improved RMSE on two disjoint holdouts (2023-25 and 2021-22), so off by default.
# Set True to restore them.
KEEP_OWN_EPA_WPA = False

# --- Schedule / travel context ------------------------------------------
# Rest days (schedules ``home_rest``/``away_rest``) below this = a short week
# (e.g. a Thursday game 4 days after a Sunday game).
SHORT_WEEK_MAX_REST = 4
# Rest days at or above this = coming off a bye week.
OFF_BYE_MIN_REST = 11

# Home time zone of each team, expressed as hours BEHIND US Eastern
# (ET=0, CT=1, MT=2, PT=3). Used to compute body-clock travel shift.
TEAM_TZ = {
    # Eastern
    "ATL": 0, "BAL": 0, "BUF": 0, "CAR": 0, "CIN": 0, "CLE": 0, "DET": 0,
    "IND": 0, "JAX": 0, "MIA": 0, "NE": 0, "NYG": 0, "NYJ": 0, "PHI": 0,
    "PIT": 0, "TB": 0, "WAS": 0,
    # Central
    "CHI": 1, "DAL": 1, "GB": 1, "HOU": 1, "KC": 1, "MIN": 1, "NO": 1, "TEN": 1,
    # Mountain (Arizona keeps MST year-round; treated as Mountain here)
    "DEN": 2, "ARI": 2,
    # Pacific
    "LA": 3, "LAR": 3, "LAC": 3, "LV": 3, "SEA": 3, "SF": 3,
    # Legacy abbreviations that may appear in older data
    "OAK": 3, "SD": 3, "STL": 1,
}

# International neutral-site stadiums -> tz offset (hours behind ET; negative =
# ahead of ET). Matched by keyword in the schedules ``stadium`` name. Domestic
# neutral sites (Super Bowls, relocations) fall back to the home team's tz.
INTL_STADIUM_TZ = {
    "tottenham": -5, "wembley": -5,                 # London
    "allianz": -6, "deutsche bank": -6,             # Germany (Munich/Frankfurt)
    "azteca": 1,                                     # Mexico City
    "corinthians": -2,                              # Sao Paulo, Brazil
}

# --- Advanced pbp-derived rates -----------------------------------------
# Explosive play thresholds (yards gained).
EXPLOSIVE_PASS_YARDS = 20
EXPLOSIVE_RUSH_YARDS = 10
# A drive "reached the red zone" when a play snapped inside this yardline_100.
REDZONE_YARDLINE = 20

# Injury statuses that count a depth-chart starter as OUT for the "starters out"
# count (mirrors the QB logic).
STARTER_OUT_STATUSES = ("Out", "Doubtful")

# Weather substrings (in the pbp ``weather`` text) that flag precipitation.
PRECIP_KEYWORDS = ("rain", "snow", "shower", "sleet", "drizzle", "flurr", "wintry")

# --- XGBoost hyper-parameters (tuned in the notebook) --------------------
XGB_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.02,   # 0.02 beat 0.01 on both walk-forward holdouts (2024, 2025)
    max_depth=2,
    min_child_weight=10,
    subsample=0.7,
    colsample_bytree=0.5,
    gamma=0.6,
    reg_alpha=1e-4,
    reg_lambda=2.0,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
)
