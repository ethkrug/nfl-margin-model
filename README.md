# NFL Point-Margin Model

A machine-learning pipeline that predicts the **final point margin of NFL games**
from play-by-play data. It pulls the nflverse feeds with
[`nfl_data_py`](https://github.com/nflverse/nfl_data_py), engineers
opponent-adjusted team, quarterback, schedule and situational features, trains a
tuned XGBoost regressor, and ships a self-contained web app that shows each
week's projected lines. Team form is a slow-decaying average that runs across
seasons but carries last season over according to how much of the team is still
the same: its quarterback, its head coach, and how much each stat persists.

Everything is evaluated **walk-forward** — for each test season the model is
trained only on the seasons before it — so no result on this page comes from a
model that saw its own test data.

## Results

Walk-forward holdouts, at the team-game level (`python -m nfl_margin_model.evaluate`):

| Holdout season | Trained on | n | RMSE | Mean abs. error | Straight-up winners |
|---|---|---|---|---|---|
| 2024 | 2010–2023 | 570 | 12.81 | 9.93 pts | 69.6% |
| 2025 | 2010–2024 | 570 | 12.75 | 10.04 pts | 63.3% |
| **Pooled** | — | **1,140** | **12.78** | **9.98 pts** | **66.5%** |

168 features, 285 games per season, each contributing one row per team.

Earlier versions of this page reported 12.68 pooled. That number was flattered
by a feature that leaked the result: the depth-chart rank of the opponent QB
who *actually* took the most snaps, known only after kickoff. Removing it cost
0.08 RMSE (see [Leakage safety](#leakage-safety)); every number here is without
it.

**2024 and 2025 are not a fully clean test of this model.** Several older
settings were chosen by looking at them (the learning rate, the prior-season
blend weights, which redundant columns to drop), which tilts these two seasons
toward the configuration they were tuned on. The current team-form features
were chosen on 2014–2023 instead and then checked once on 2024/2025 and on the
2026 games played so far, which nothing was tuned on:

| Team form | 2014–2023 (chosen on) | 2024/2025 | 2026 weeks 1–8 (120 games) |
|---|---|---|---|
| 5- and 10-game windows (previous) | 13.305 | **12.73** | 12.45 |
| Slow decay across seasons, no continuity | **13.244** | 12.85 | 12.55 |
| **Continuity-weighted decay (shipped)** | 13.256 | 12.78 | **12.37** |

RMSE; 2014–2023 is each season predicted by a model trained on every earlier
season, averaged over five seeds. See
[Continuity-weighted team form](#continuity-weighted-team-form) for why the
plain decay breaks after 2023 and what the continuity terms fix.

### Reproducing these numbers

The holdout table at the top and the scorecard below were produced by running
`evaluate` and `scorecard` against live nflverse data on the current commit, not
carried over from an earlier version of the model. The comparison tables (team
form, continuity, rejected ideas) come from experiments run on the same data
with the same walk-forward protocol. Runs are deterministic: XGBoost is pinned
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
| With Pro Bowl data | 12.81 / 12.75 / **12.78** | 9.98 | 66.5% |
| Without (fresh clone) | 12.90 / 12.79 / **12.84** | 10.02 | 66.5% |

So a clone lands **0.06 RMSE worse** pooled, and picks **exactly the same
winners** — the Pro Bowl correction is gated so it can sharpen a margin but never
flip which team is favoured, which is why that column is identical rather than
merely close. The holdout and scorecard numbers otherwise reproduce exactly.

The market's own closing spread is **not** an input to the model. It appears in
the frontend and the in-season scorecard only, as something to compare against.

### In-season scorecard

`python -m nfl_margin_model.scorecard` grades the games already played this
season week by week: the model's line and the market's closing spread, both
against the actual margin. Through 2026 week 8 (120 games):

| | MAE | RMSE | Straight-up winners |
|---|---|---|---|
| Model | 9.71 | 12.37 | 60.5% |
| Market closing spread | 9.48 | 12.09 | 66.4% |

The model's line finished closer to the result than the market's in 40% of
games. It takes `--cache`, or `--preds` to score an existing payload without
retraining, and `--csv` to write the table.

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

# build the web app with the latest games
python -m nfl_margin_model.frontend.generate --no-logos
```

**Use `--no-logos`.** It is the build that works anywhere: it needs no image
library and fetches no images, so the only thing it downloads is the game data
itself. Dropping the flag gives you the same page with each club's logo
embedded, which additionally needs Pillow and a reachable ESPN — see
[The web app](#the-web-app) for why the logos are a local-only extra.

`evaluate` and the frontend both go through `predict.py`; `run_pipeline.py` is
the readable narration of the same feature engineering, split out so the stages
can be followed one at a time. The two now share a depth-chart loader, so they
see the same features.

The first run downloads ~15 seasons of play-by-play from nflverse and takes a
few minutes. Add `--save-cache <dir>` to keep the raw feeds, and every later run
can take `--cache <dir>` to read them off disk instead — nearly instant, and no
network at all. Both `evaluate` and `generate` accept the pair; `*.parquet` and
`.cache/` are already gitignored.

```bash
python -m nfl_margin_model.frontend.generate --no-logos --save-cache .cache/nflverse
python -m nfl_margin_model.frontend.generate --no-logos --cache .cache/nflverse
```

Requires Python 3.11+.

---

## How it works

The pipeline is a straight line through `nfl_margin_model/pipeline.py`:

| Stage | Module | What happens |
|-------|--------|--------------|
| 1 · Load | `data.py` | play-by-play, weekly, depth charts, injuries, schedules (2010–present) |
| 2 · Game frame | `features.py` | collapse plays to game totals; build the target and cleaned game context |
| 3 · Team-games | `features.py`, `advanced.py` | reshape to one row per team-game with `for_`/`allow_` splits; add red-zone, explosive-play, pressure, kicking and fumble rates |
| 4 · Team form | `features.py`, `continuity.py` | **opponent-adjust** EPA; a 5-game window for recent form, plus a continuity-weighted decay average over the last two seasons |
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

**Prior-season blending in the 5-game window, week-dependent.** In the first
weeks of a season the window is mostly empty; the empty slots are seeded from
last season's form. The right weight differs by phase — weeks 2–4 have a 1–3
game sample and want a strong prior (4×), while from week 5 on the current season
is reliable and a light prior (1×) wins. The prior falls out entirely once the
window fills.

**The quarterback is the one the depth chart named, not the one who played.**
Which QB ends up taking the most dropbacks is only knowable after kickoff, so
choosing him to represent the team leaks the game into its own feature. The
published depth chart answers the same question from information available
beforehand, and the injury report handles the case where the named starter is
ruled out. A QB1 who is named but does not play still has his form fetched, as
of the state left by his own most recent start.

Building each QB's *history* from whoever actually threw is a different matter
and stays as it is — those box scores already exist when the current game kicks
off, and a game's own EPA never feeds its own feature.

This costs accuracy and is worth it: **+0.15 pooled RMSE and −1.6pp on winners**
against selecting by dropbacks. Two features went with it — `qb_depth_order` and
`is_designated_starter` are both computed from who actually played, so neither
can be a model input; they survive as diagnostics. What remains is what a real
pre-game forecast could have used, so the numbers at the top are lower than they
would otherwise be, and they are the ones that would hold up live.

The pre-game version of that depth signal *is* a feature, for both teams
(`qb_depth_pregame`, `opp_qb_depth_pregame`). Take the week's published depth
chart (or, with no chart, the team's QBs from its previous game ranked by
dropbacks), remove anyone the injury report rules out, and the first QB left is
the expected starter. His place on that list is the feature. It names the QB who
actually started in 90.8% of 2010–2025 team-games, against 87.7% for the
chart's QB1, and is right in 79% of the 335 games where QB1 was ruled out.

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
isolate among 168 features — fed in as tree inputs, those columns are
null-to-harmful. So the residual is removed explicitly instead, with a linear
slope fit on **out-of-fold** residuals pooled across many seasons. It is gated
(small imbalances are left untouched) and may never flip which team is favoured.
Walk-forward, it helps in test seasons.

### Continuity-weighted team form

The 10-game window that used to supply the longer view stopped at the season
boundary. A longer memory looked clearly better on 2014–2023 (55 decay curves
tried, nearly all of them better than the windows) and then lost on 2024, 2025
and 2026. The reason is in the data: how well a team's net EPA per play matches
its previous season's fell from 0.43–0.52 (2018–2023) to 0.28, 0.32 and 0.33.
Teams have turned over more, and the kickoff rule changes moved whole stats
(touchbacks went from 3.3–4.0 a team-game to 1.4 in 2025 and 1.2 in 2026). A
memory that treats last season as this season's team misfires in those years.

How much a team carries over depends on what changed. Season-to-season
correlation of EPA per play, 2011–2023:

| Offseason change | Team-seasons | Offense | Defense | Net |
|---|---|---|---|---|
| Kept QB and head coach | 216 | 0.46 | 0.26 | 0.42 |
| Same QB, new coach | 52 | 0.30 | 0.30 | 0.33 |
| New QB, same coach | 99 | 0.14 | 0.29 | 0.31 |
| New QB and new coach | 49 | −0.27 | 0.17 | −0.10 |

So `continuity.py` keeps the long memory but carries earlier seasons over by
continuity. Each stat gets one average over the team's previous 34 games across
up to two seasons, weighted by a linear fade (the last game counts 1, then 0.97,
0.94, …, about 0.53 a season back), and every earlier season's game is:

- **measured against its own season's league level** and re-centred on this
  season's, so rule changes do not carry across years;
- **shrunk by that stat's season-to-season carryover**;
- **shrunk further by continuity**: offense stats by whether the QB and head
  coach are the same, defense and other stats by the coach. Within a season, a
  game played by a different QB than this week's expected starter is shrunk too.

Every carryover estimate for a season is measured only from the seasons before
it, the QB is this week's pre-game expected starter, and the coach comes from the
schedule. Rebuilding with every feed cut off after 2024 week 9 reproduces the
week-10 features exactly. Shrinking the carried level beat down-weighting it: an
extra offseason weight of 0.7 or 0.4, or 0.5 when the QB or coach changed, was
worse. A weighted average cannot use a down-weight to move a team's week-1
level, since all its games are from last season either way, whereas shrinking
can.

### Leakage safety

Every feature is built from information available before kickoff, and this is the
constraint the code is most careful about:

- all rolling stats are shifted, so a game never sees itself;
- 2025+ depth charts are daily snapshots rather than weekly rows, so they are
  date-mapped: a week's chart is the last snapshot taken before that week's
  **first** kickoff, and snapshots taken after the season's final kickoff are
  discarded rather than folded into the last week. Verified against the 2025
  schedule, every week;
- a week's chart is that team's latest snapshot taken **whole**, not the latest
  row per player. A chart is a document: a player who is no longer on it has
  left. Merging rows per player instead makes a departure invisible — the
  player's last row on the old club is never superseded, because his later rows
  carry the new one — which across an offseason (where every snapshot collapses
  onto week 1) left teams listing several traded-away quarterbacks at rank 1
  next to the real starter;
- Pro Bowl pedigree for a season is drawn only from **prior** seasons;
- everything *fitted while building the frame* — the replacement-level QB
  baseline and the advanced-metric fill — is bounded by one `fit_through`: its
  own `train_through` for a single fit, the earliest holdout minus one for the
  walk-forward evaluation, which builds one frame spanning every fold;
- undefined rate metrics (no FG attempt → no FG%, no red-zone trip → no
  red-zone TD rate; 1,977 cells) are imputed from the **team's own previous 10
  games**, shifted so a game never informs its own fill, falling back to the
  bounded league mean only where a team has no prior reading. Missing weather
  (200 of 4,364 games) likewise comes from that venue's **earlier** games — 199
  covered by the venue's own history, 1 by the league's prior mean. Both were
  pooled means over the whole frame, which let a game already played this week
  shift the values used by the rest of that week;
- the projected upcoming week is purely additive — verified by building the
  frame both ways and diffing: appending it changes **0 feature cells across all
  8,694 already-played team-game rows**, and only adds two rows per pending game
  (checked with a week half-played, the case where a projected row sits in the
  same week as a completed one);
- the pipeline is *future-blind* end to end — verified by deleting every
  game after a cut point and rebuilding: **0 of 1,478,652 cells** change across
  203 columns on the 7,284 earlier rows;
- the QB representing a game is the one named pre-game, never the one who turned
  out to play most (see above), and the features derived from who actually
  played are excluded from the model — including the opponent's copy of
  `qb_depth_order`, which `add_matchup_features` used to pass through (it keeps
  every `opp_qb*` column). That leak was the 8th most influential feature and
  was worth 0.08 RMSE on 2024/2025; its pre-game replacement is
  `opp_qb_depth_pregame`;
- the continuity-weighted decay uses only games before the one being built, and
  each season's carryover and continuity estimates come only from earlier
  seasons. Verified by cutting every feed off after 2024 week 9 and rebuilding,
  so week 10 is projected as an upcoming week: its decay features match the
  full-data build exactly (0 mismatches across 67 features × 28 team-games);
- depth-chart club codes are normalised to the current franchise codes before
  any join. nflverse keeps the historical code (`OAK`/`SD`/`STL`) while
  play-by-play uses the current one, so an unmapped join silently misses 374
  team-games and hands three franchises a decade of neutral QB and injury
  features;
- every ordering that selects a row (designated starter, latest depth-chart
  snapshot, primary starter) is a *total* order under a stable sort. Ranking on
  a column with ties and letting the sort break them means the winner depends on
  incidental array layout — in this codebase that was live, and unrelated rows
  in one season could silently reassign the starting quarterback in another (see
  [Reproducibility](#reproducing-these-numbers)).

---

## The model

A deliberately small, heavily-regularized XGBoost regressor — 168 features on
~7,000 team-games rewards restraint over capacity:

```python
n_estimators=300, learning_rate=0.02, max_depth=2, min_child_weight=10,
subsample=0.7, colsample_bytree=0.5, gamma=0.6, reg_lambda=2.0
```

Fixed `random_state` with `tree_method="hist"` makes runs bit-identical, across
processes as well as within one — which is what makes small tuning deltas
readable as real rather than as run-to-run noise.

The size of the trees was re-checked without the leaked feature: depth 1–6,
learning rate 0.01 / 0.02 / 0.05 and 50–3,000 trees on the 2016–2023 holdouts,
leaders re-run on five seeds. This setting sits on a flat optimum; every depth
1–3 setting with enough trees landed within seed noise of it, and deeper or
longer runs overfit.

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
| Head-coach continuity / tenure / staleness as features | Null across the board — though as a modifier on how much last season carries over, coach continuity is part of the shipped decay |
| Roster churn (Pro Bowl arrivals and departures) | Below the noise floor |
| Stronger QB shrinkage (K = 5, 7, 10, 14) | Degrades monotonically; K = 3 is optimal |
| Per-week tuning of the prior-season weight | Total spread across 7 schedules is 0.03 RMSE — no leverage |
| A live weather-forecast integration | *Perfect* weather ties climatology (12.761 vs 12.759); weather moves totals much more than margins |
| Sourcing real offseason QB changes for accuracy | Actual, most-recent and primary starters land within 0.1 RMSE on Week-1 openers — a credibility feature, not an accuracy one |
| A 15-game window (5/10/15, or 5/15) | Better on 2014–2023, worse on 2024/2025 |
| A plain slow decay across seasons (55 curve shapes: power, hyperbolic, exponential, linear, flat) | Best on 2014–2023, clearly worse on 2024–2026 when team strength carried over less |
| Extra offseason down-weighting (0.7 or 0.4 per season; 0.5 on a QB or coach change) | Worse than shrinking the carried level by continuity |
| Recency-weighted training (season half-life 1–12, or none) | No half-life beat equal weights; short ones degrade steadily |
| A two-stage play-level ratings model (weighted ridge on every play rating team offense, defense and QB, then XGBoost) | About even season by season through 2023 (better in 5 of 12 seasons 2014–2025), worse on 2024/2025 (12.96 vs 12.73) |
| True offense / defense EPA from the plays | Accuracy unchanged (13.318 vs 13.310 on 2014–2023; 12.725 vs 12.727 on 2024/2025) — see below |

The recurring lesson: early-season corrections must shut off promptly, and a
holdout season is small enough (~0.35 RMSE standard error) that anything under
about a tenth of a point is noise no matter how good the story is. Two holdout
seasons cannot resolve a difference under ~0.08 RMSE either, which is why changes
are now chosen on ten seasons (2014–2023) before 2024/2025 are touched.

**A known limitation, kept on purpose.** The `for_` / `allow_` EPA columns come
from nflverse's `total_home_*` / `total_away_*` running totals, which are
zero-sum: every play adds its EPA to one side and subtracts it from the other.
So each team's "allowed" EPA is exactly minus its "for" EPA, both are its net
EPA, and `edge_off_*`, `edge_def_*` and `edge_net_*` are the same signal (the
game card's "Offense edge" and "Defense edge" are mirror images). Rebuilding
offense and defense from the plays makes them genuinely independent (correlation
+0.07 instead of −1.00) but leaves accuracy where it is: the net edge is
identical either way, and for a final margin the split adds nothing the model
can use.

---

## The web app

`nfl_margin_model/frontend/` builds a **single self-contained HTML file** —
predictions, both typefaces and (in the local build) the club logos all embedded,
zero external requests. It opens on any machine, offline, with no install. Each week is a ledger of games
showing the model's line, the win-probability split, the market line, and an
expandable per-game panel.

It builds in two variants. **Start with the first one** — it is the one that
works on any machine, and the one whose output is checked in so you can see what
you are building before you build it:

```bash
# start here: team-colour tiles with abbreviations, no club marks.
# no Pillow, no image downloads -- just the game data.
python -m nfl_margin_model.frontend.generate --no-logos

# optional, local only: the same page with each club's logo embedded.
# needs Pillow and fetches 32 logos from ESPN on first run.
python -m nfl_margin_model.frontend.generate
```

Only the `--no-logos` build is tracked here
([`index_no_logos.html`](nfl_margin_model/frontend/index_no_logos.html)). NFL
club logos are trademarks of their owners and not this project's to
redistribute, so the logo build stays local and fetches its own copies — see
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md). It is the same design either
way: each team gets a tile in its own colour, carrying the club logo when one is
available and the team abbreviation when it is not.

The published variant needs no image dependencies at all, so it builds from a
fresh clone with nothing but the tracked font cache. Once built, send the file
to someone, or host it alongside its own payload (`index_no_logos.html` +
`preds_no_logos.json`) on any static host so recipients pull your latest numbers
on refresh. Full details in
[`nfl_margin_model/frontend/README.md`](nfl_margin_model/frontend/README.md).

`TRAIN_THROUGH` in `predict.py` controls the training window — the model trains on
every season through it and displays the season after as pure hold-out. **Bump it
once a year**, after that season ends.

Within the season, nothing needs bumping. Each build fetches the live season's
feeds alongside the training history and projects the earliest week that still
has games left, so a rebuild picks up whatever has been played since the last
one: finished games appear with their results and their (genuinely out-of-sample)
predictions, the rolling windows advance onto them, and the next week is
projected off them. A week that is only partly played is handled per game — the
Thursday game shows a final score while the Sunday slate is still predicted.

The live season is fed in as *history*, never as training data: it is on the
display side of `TRAIN_THROUGH`, so a week-1 result sharpens the week-2 features
without the model ever fitting on it.

---

## Repo layout

```
run_pipeline.py               entry point
requirements.txt              runtime dependencies
LICENSE                       MIT, covering the source code
THIRD-PARTY-NOTICES.md        fonts (bundled, OFL) and club logos (not bundled)
licenses/OFL-1.1.txt          the font licence itself
nfl_margin_model/
  config.py                   every tunable knob, with the reasoning for each
  fetch.py                    nflverse downloads: retries + season-coverage checks
  data.py                     nflverse loaders + schema unification
  features.py                 game/team-game features, opponent adjustment, rolling
  continuity.py               continuity-weighted decay averages (team form)
  advanced.py                 red zone, explosive plays, pressure, kicking
  qb.py                       starter identification, QB quality, shrinkage
  schedule.py                 rest, travel, starters out, Pro Bowl absences
  weather.py                  stadium coordinates, forecast, climatology
  projection.py               shell rows for upcoming, unplayed games
  model.py                    targets, preprocessing, split, training, PB correction
  predict.py                  single source of truth for predictions
  evaluate.py                 walk-forward holdout evaluation
  scorecard.py                in-season scorecard: model and market vs results
  pipeline.py                 orchestration
  console.py                  dependency-free terminal formatting
  frontend/                   self-contained web app
    template.html             the UI; edit this to change the design
    generate.py               builds a page (--no-logos for the tracked variant)
    render_static.py          JS-free variant of the same page
    index_no_logos.html       the tracked, publishable build
    preds_no_logos.json       its prediction payload
    fonts_cache.json          the two OFL typefaces, base64'd
  pro-bowl/                   Pro Bowl exports (data not tracked — see its README)
```

The logo-bearing builds (`index.html`, `index_offline.html`, `preds.json`,
`logos_cache.json`) and the Pro Football Reference exports are generated or
supplied locally and are deliberately untracked; see
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).

`config.py` is worth reading on its own: each constant carries the backtest that
set it, including the ones that argue against changing it.

---

## Data and caveats

Data comes from [nflverse](https://github.com/nflverse) via `nfl_data_py`
(play-by-play, weekly, depth charts, injuries, schedules) and, for Pro Bowl
selections, from manual Pro-Football-Reference exports.

Those feeds are fetched over HTTPS at run time and occasionally fail on a
dropped connection. `fetch.py` wraps every one of them: transient failures are
retried with backoff, and a season that still never arrives raises rather than
being skipped. That second half matters — `nfl_data_py.import_pbp_data` catches
its own per-season errors, prints `Data not available for 2011` and returns a
frame with that season missing, which would otherwise train the model on a hole
in its history and report success.

**Those PFR exports are not included in this repo** — they aren't mine to
redistribute. The pipeline detects their absence, says so, and runs normally with
every `pb_*` feature zeroed; see
[`nfl_margin_model/pro-bowl/README.md`](nfl_margin_model/pro-bowl/README.md) for
the expected format.

This is a modeling project, published for analysis and as a portfolio piece. It
is **not wagering advice**: predictions are graded against actual game margins.
The market's closing spread is shown beside them only as a benchmark, and no
part of the project is designed or validated for placing bets.

## License

MIT — see [LICENSE](LICENSE). That covers the source code in this repository.

The build pulls in third-party material the MIT license cannot speak for: two
OFL-licensed typefaces, which are bundled with their notices, and NFL club
logos, which are deliberately not redistributed here at all. What arrives on
what terms is set out in
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).
