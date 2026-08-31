# NFL Prediction Model — frontend

A self-contained web app for the NFL point-margin model. Each week's games are
shown as a ledger: the model's line, win-probability split, the market line and
the edge between them, with an expandable panel per game (model vs market,
projected-margin scale, game info).

## Files

The page builds in two variants. `generate.py --no-logos` produces
`index_no_logos.html` + `preds_no_logos.json`, which carry no club marks (just
team-colour tiles with abbreviations) and **are tracked here**. Plain
`generate.py` produces the logo build, which is **not tracked** — NFL club logos
are not ours to redistribute (see
[THIRD-PARTY-NOTICES.md](../../THIRD-PARTY-NOTICES.md)).

Files marked **(built, untracked)** below are the logo variant; run the refresh
command to produce them locally, and note the first such run needs network
access and Pillow.

| File | What it is |
|------|-----------|
| `index.html` | **(built, untracked) The deliverable.** One self-contained file — open in any browser. No install, no server, no internet. This is what you send people. |
| `template.html` | The app UI (a `/*__DATA__*/` placeholder for predictions, `/*__FONTS__*/` for the embedded typefaces). **Edit this to change the design.** |
| `generate.py` | The **refresh command**. Runs the model and bakes predictions into the page. `--no-logos` builds the tracked, publishable variant. |
| `index_no_logos.html` | **(tracked)** The publishable build — same page, team-colour tiles instead of club marks. |
| `preds_no_logos.json` | **(tracked)** Its prediction data. |
| `preds.json` | **(built, untracked)** The prediction data alone. Only needed for the hosted auto-refresh mode below. |
| `logos_cache.json` | **(built, untracked)** Team logos, downscaled and base64'd. Fetched once, then reused. Untracked: NFL club marks. |
| `fonts_cache.json` | Archivo + JetBrains Mono woff2 (latin subset), base64'd. Fetched once, then reused. Tracked: both are OFL-licensed and bundled with their notices. |

## Refreshing the predictions (you run this)

```bash
# from the project root (the folder containing the nfl_margin_model package).
# start here -- works anywhere, no image library, no image downloads:
python -m nfl_margin_model.frontend.generate --no-logos

# optional, local only: same page with club logos embedded.
# needs Pillow, and fetches 32 logos from ESPN on first run:
python -m nfl_margin_model.frontend.generate

# instant rebuild from cached parquet, for development (either variant):
python -m nfl_margin_model.frontend.generate --no-logos --cache <dir>
```

Both pull the latest games via `nfl_data_py` and retrain. `--no-logos` rewrites
`index_no_logos.html` + `preds_no_logos.json` (the tracked, publishable pair);
without the flag it rewrites `index.html` + `preds.json`, which are untracked.

## Sharing it — no internet required

Either build is **completely self-contained**: predictions, typefaces (and the
club logos, in the logo build), all CSS and JS are embedded in the file. There
are zero external references — no CDN, no fonts server, no analytics. It renders
identically on a machine that has never been online.

Below, "the page" means whichever you built — `index_no_logos.html` or
`index.html`. **Only the `--no-logos` build is yours to publish**; the logo build
carries club marks and is for your own use (see
[THIRD-PARTY-NOTICES.md](../../THIRD-PARTY-NOTICES.md)).

**A. Send the file (snapshot).** Email/AirDrop/Slack/USB the page. Recipients
double-click it; it opens in any browser. Nothing else needed.
- Some mail clients block `.html` attachments — zip it, or send it via a file
  share. In Gmail/Drive, "preview not available" is normal: click **Download**.
- It shows whatever was current when you generated it. To update someone, re-run
  `generate.py` and send the new file.

**B. Host it (recipients refresh for your latest).** Put the page **and its own
payload** on any static host (Netlify, S3, an internal share) — that is
`index_no_logos.html` + `preds_no_logos.json`, or `index.html` + `preds.json`.
Each build fetches the payload it was built against (the filename is baked in at
build time), so the two variants never cross-load; keep the matching pair
together. Re-upload the payload to update everyone.
This is the *only* mode that touches a network, and it is optional — that fetch
failing (as it does on `file://`) simply leaves the embedded predictions in
place.

> Recipients can never re-run the *model* — that needs Python and the data feed.
> "Refresh" in mode B means "pull your latest published numbers."

## Training window (bump this each year)

`TRAIN_THROUGH` lives in `nfl_margin_model/predict.py` (currently **2025**). The
model trains on every season through it and displays the season after as pure
hold-out — so today it trains through 2025 and shows **2026**. Override per-run:

```bash
python -m nfl_margin_model.frontend.generate --train-through 2025
```

The new season also has to be available to `nfl_data_py` and inside
`config.PBP_YEARS`.

## Notes

- Predictions are **out-of-sample** (seasons later than `TRAIN_THROUGH`).
- The model line is graded against the **actual game margin**, not against the
  market spread. The market line is shown for comparison only and is not an
  input to the model. For analysis, not wagering advice.
- Weeks with no games yet are dimmed in the rail and show `—`.
- Injury-driven adjustments (the Pro Bowl absence correction) only engage once
  weekly injury reports exist for the season being predicted.
