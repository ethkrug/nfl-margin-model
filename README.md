# NFL Point-Margin Model

A machine-learning pipeline that predicts the **final point margin of NFL games**
from play-by-play data. It pulls the nflverse feeds with
[`nfl_data_py`](https://github.com/nflverse/nfl_data_py), engineers
opponent-adjusted team, quarterback, schedule and situational features, trains a
tuned XGBoost regressor, and ships a self-contained web app that shows each
week's projected lines.

Everything is evaluated **walk-forward** — for each test season the model is
trained only on the seasons before it — so no result on this page comes from a
model that saw its own test data.

## Results

Walk-forward holdouts, at the team-game level (`python -m nfl_margin_model.evaluate`):

| Holdout season | Trained on | n | RMSE | Mean abs. error | Straight-up winners |
|---|---|---|---|---|---|
| 2024 | 2010–2023 | 570 | 12.61 | 9.83 pts | 70.9% |
| 2025 | 2010–2024 | 570 | 12.45 | 9.80 pts | 66.1% |
| **Pooled** | — | **1,140** | **12.53** | **9.82 pts** | **68.5%** |

169 features, 285 games per season, each contributing one row per team.

### Reproducing these numbers

Every figure in this README was produced by running the command above against
live nflverse data on the current commit, not carried over from an earlier
version of the model. Runs are deterministic: XGBoost is pinned
(`random_state=42`, `tree_method="hist"`) and every sort that picks a row is a
total order under a stable sort, so the same inputs give bit-identical output
across processes. That property is load-bearing — it is what makes a 0.05 RMSE
difference readable as a real effect rather than as run-to-run jitter.

One input is **not** in this repo: the Pro Bowl selections (see
[Data and caveats](#data-and-caveats)). Without them the pipeline still runs —
every `pb_*` feature is simply zero — but the results are slightly worse, and
these are the numbers a fresh clone actually produces:

| | RMSE (2024 / 2025 / pooled) | MAE | Winners |
|---|---|---|---|
| With Pro Bowl data | 12.61 / 12.45 / **12.53** | 9.82 | 68.5% |
| Without (fresh clone) | 12.67 / 12.49 / **12.58** | 9.84 | 68.5% |

So a clone lands **0.05 RMSE worse** pooled, and picks **exactly the same
winners** — the Pro Bowl correction is gated so it can sharpen a margin but never
flip which team is favoured, which is why that column is identical rather than
merely close. Everything else in this README reproduces exactly.

The market's own closing spread is **not** an input to the model. It appears in
the frontend only, as something to compare against.

---

## Quickstart

```bash
git clone https://github.com/ethkrug/nfl-margin-model.git
cd nfl-margin-model
pip install -r requirements.txt

# walk-forward holdout evaluation -- the authoritative numbers above
python -m nfl_margin_model.evaluate

# the narrated end-to-end pipeline, for reading the stages
python run_pipeline.py          # or: python -m nfl_margin_model

# rebuild the web app with the latest games
python -m nfl_margin_model.frontend.generate
```

`evaluate` and the frontend both go through `predict.py`; `run_pipeline.py` is
the readable narration of the same feature engineering, split out so the stages
can be followed one at a time. The two now share a depth-chart loader, so they
see the same features.

The first run downloads ~15 seasons of play-by-play from nflverse and takes a
few minutes. Both `evaluate` and `generate` accept `--cache <dir>` to read
previously-saved parquet instead, which makes iteration nearly instant.

Requires Python 3.11+.

---

## How it works

The pipeline is a straight line through `nfl_margin_model/pipeline.py`:

| Stage | Module | What happens |
|-------|--------|--------------|
| 1 · Load | `data.py` | play-by-play, weekly, depth charts, injuries, schedules (2010–present) |
| 2 · Game frame | `features.py` | collapse plays to game totals; build the target and cleaned game context |
| 3 · Team-games | `features.py`, `advanced.py` | reshape to one row per team-game with `for_`/`allow_` splits; add red-zone, explosive-play, pressure, kicking and fumble rates |
| 4 · Rolling | `features.py` | **opponent-adjust** EPA, roll every stat over prior games, blend in the prior season early |
| 5 · QB | `qb.py` | identify the designated starter, rate them, handle injuries and backups |
| 6 · Schedule | `schedule.py` | rest, bye, short weeks, travel, starters out, Pro Bowl absences |
| 7 · Matchup | `features.py` | join the opponent's profile and build the `edge_*` differentials |
| 8 · Train | `model.py` | one-hot the categoricals, split by season, fit XGBoost, apply the Pro Bowl correction |

The unit of prediction is a **team-game**: each game contributes two rows, one
from each side's perspective, and the two are antisymmetric by construction.

### The features that matter

An ablation over the whole feature set (train ≤2023, validate 2024, test 2025)
found one group doing nearly all of the work. (This is a fixed-split
ablation, not the walk-forward run above, so the numbers are not directly
comparable to the table at the top — only to each other.)


| Configuration | Test RMSE | R² | Winners |
|---------------|-----------|-----|---------|
| Own-team rolled stats + QB only (no opponent) | 13.56 | 0.10 | 62% |
| + opponent strength and `edge_*` differentials | **12.87** | **0.19** | **65%** |
| + defensive opponent-adjustment (shipped) | 12.83 | 0.193 | 65% |

Without opponent information the model is nearly blind: each row sees only one
team. Adding the matchup roughly **doubles R²**. Schedule, advanced play-by-play
rates and weather are each about neutral on a single test season — NFL-sound,
not harmful, kept, but they are not what makes this work.

### Things the model does that are less obvious

**Opponent adjustment.** A team's EPA is credited against the quality of the
defenses it faced, so a hot start against a soft schedule doesn't read as
strength.

**Prior-season blending, week-dependent.** In the first weeks of a season a
rolling window is mostly empty; the empty slots are seeded from last season's
form. The right weight differs by phase — weeks 2–4 have a 1–3 game sample and
want a strong prior (4×), while from week 5 on the current season is reliable and
a light prior (1×) wins. The prior falls out entirely once the window fills.

**Quarterback rating with shrinkage.** Starters come from the weekly depth chart,
cross-checked against the injury report, so a ruled-out QB1 correctly hands the
row to the backup. A QB's rolling EPA is shrunk toward a replacement-level
baseline (fit on training seasons only), which stops a three-start rookie from
being rated like a franchise arm.

**Offseason regression, at openers only.** A QB's form carries across the
offseason without resetting, so someone who ended the year hurt or slumping
starts the new one underrated. At a QB's season opener — and *only* there — their
quality is pulled 75% toward their career baseline, fading to exactly zero by
their second start. The auto-shutoff is the whole reason it works; an earlier
version that kept regressing through five starts damaged the mid-season weeks.

**Pro Bowl absence correction.** Teams missing players with Pro Bowl pedigree
underperform the model by a real margin (~2.4 points when two or more are out),
but the situation is far too rare (~6% of team-games) for a depth-2 ensemble to
isolate among 169 features — fed in as tree inputs, those columns are
null-to-harmful. So the residual is removed explicitly instead, with a linear
slope fit on **out-of-fold** residuals pooled across many seasons. It is gated
(small imbalances are left untouched) and may never flip which team is favoured.
Walk-forward, it helps in test seasons.

### Leakage safety

Every feature is built from information available before kickoff, and this is the
constraint the code is most careful about:

- all rolling stats are shifted, so a game never sees itself;
- 2025+ depth charts are daily snapshots rather than weekly rows, so they are
  date-mapped: a week's chart is the last snapshot taken before that week's
  **first** kickoff, and snapshots taken after the season's final kickoff are
  discarded rather than folded into the last week. Verified against the 2025
  schedule, every week;
- Pro Bowl pedigree for a season is drawn only from **prior** seasons;
- the replacement-level QB baseline is fit on train-era games only;
- the projected upcoming season is purely additive — verified that appending it
  changes 0 of 360 already-played feature rows;
- every ordering that selects a row (designated starter, latest depth-chart
  snapshot, primary starter) is a *total* order under a stable sort. Ranking on
  a column with ties and letting the sort break them means the winner depends on
  incidental array layout — in this codebase that was live, and unrelated rows
  in one season could silently reassign the starting quarterback in another (see
  [Reproducibility](#reproducing-these-numbers)).

---

## The model

A deliberately small, heavily-regularized XGBoost regressor — 169 features on
~7,000 team-games rewards restraint over capacity:

```python
n_estimators=300, learning_rate=0.02, max_depth=2, min_child_weight=10,
subsample=0.7, colsample_bytree=0.5, gamma=0.6, reg_lambda=2.0
```

Fixed `random_state` with `tree_method="hist"` makes runs bit-identical, across
processes as well as within one — which is what makes small tuning deltas
readable as real rather than as run-to-run noise.

**On the error floor.** Final margin has an irreducible spread (SD ≈ 13.5 points);
a single blocked kick or a garbage-time score moves it by more than most of the
signal in the data. So RMSE improvements are small by nature, and R² and
straight-up winner accuracy are the more honest read on whether a change helped.

---

## What was tested and rejected

Kept here because the negative results are as useful as the positive ones. Each
was implemented properly and evaluated walk-forward, not hand-waved:

| Idea | Verdict |
|------|---------|
| Opponent-adjusted QB EPA | No improvement |
| Team-era home-field advantage | No improvement |
| Special-teams features | No improvement |
| Head-coach continuity / tenure / staleness | Null across the board |
| Roster churn (Pro Bowl arrivals and departures) | Below the noise floor |
| Stronger QB shrinkage (K = 5, 7, 10, 14) | Degrades monotonically; K = 3 is optimal |
| Per-week tuning of the prior-season weight | Total spread across 7 schedules is 0.03 RMSE — no leverage |
| A live weather-forecast integration | *Perfect* weather ties climatology (12.761 vs 12.759); weather moves totals much more than margins |
| Sourcing real offseason QB changes for accuracy | Actual, most-recent and primary starters land within 0.1 RMSE on Week-1 openers — a credibility feature, not an accuracy one |

The recurring lesson: early-season corrections must shut off promptly, and a
holdout season is small enough (~0.35 RMSE standard error) that anything under
about a tenth of a point is noise no matter how good the story is.

---

## The web app

`nfl_margin_model/frontend/` builds a **single self-contained `index.html`** —
predictions, team logos and both typefaces embedded, zero external requests. It
opens on any machine, offline, with no install. Each week is a ledger of games
showing the model's line, the win-probability split, the market line, and an
expandable per-game panel.

You build it rather than download it:

```bash
python -m nfl_margin_model.frontend.generate
```

The built page and its caches are **not tracked in git** — they embed NFL club
logos, which are trademarks of their owners and not this project's to
redistribute (see [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md)). The build
fetches its own copies, so the first run needs network access; afterwards the
caches make it offline-repeatable.

Once built, send the file to someone, or host `index.html` + `preds.json` on any
static host so recipients pull your latest numbers on refresh. Full details in
[`nfl_margin_model/frontend/README.md`](nfl_margin_model/frontend/README.md).

`TRAIN_THROUGH` in `predict.py` controls the training window — the model trains on
every season through it and displays the season after as pure hold-out. **Bump it
once a year.**

---

## Repo layout

```
run_pipeline.py               entry point
nfl_margin_model/
  config.py                   every tunable knob, with the reasoning for each
  data.py                     nflverse loaders + schema unification
  features.py                 game/team-game features, opponent adjustment, rolling
  advanced.py                 red zone, explosive plays, pressure, kicking
  qb.py                       starter identification, QB quality, shrinkage
  schedule.py                 rest, travel, starters out, Pro Bowl absences
  weather.py                  stadium coordinates, forecast, climatology
  projection.py               shell rows for upcoming, unplayed games
  model.py                    targets, preprocessing, split, training, PB correction
  predict.py                  single source of truth for predictions
  evaluate.py                 walk-forward holdout evaluation
  pipeline.py                 orchestration
  console.py                  dependency-free terminal formatting
  frontend/                   self-contained web app
  pro-bowl/                   Pro Bowl exports (data not tracked — see its README)
```

`config.py` is worth reading on its own: each constant carries the backtest that
set it, including the ones that argue against changing it.

---

## Data and caveats

Data comes from [nflverse](https://github.com/nflverse) via `nfl_data_py`
(play-by-play, weekly, depth charts, injuries, schedules) and, for Pro Bowl
selections, from manual Pro-Football-Reference exports.

**Those PFR exports are not included in this repo** — they aren't mine to
redistribute. The pipeline detects their absence, says so, and runs normally with
every `pb_*` feature zeroed; see
[`nfl_margin_model/pro-bowl/README.md`](nfl_margin_model/pro-bowl/README.md) for
the expected format.

This is a modeling project, published for analysis and as a portfolio piece. It
is **not wagering advice**: predictions are graded against actual game margins,
never against a betting market, and no part of it is designed or validated for
placing bets.

## License

MIT — see [LICENSE](LICENSE). That covers the source code in this repository.

The build pulls in third-party material the MIT license cannot speak for: two
OFL-licensed typefaces, which are bundled with their notices, and NFL club
logos, which are deliberately not redistributed here at all. What arrives on
what terms is set out in
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).
