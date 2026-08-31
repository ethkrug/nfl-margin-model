# Pro Bowl selections (data not included)

The `pro-bowlers*.csv` files that belong in this directory are **not tracked in
git** — they are exports from [Pro-Football-Reference](https://www.pro-football-reference.com/),
whose data is not mine to redistribute. Everything else in the repo runs without
them; see "Running without it" below.

## What the pipeline expects here

One CSV per NFL season, named `pro-bowlers<SEASON>.csv` (the year in the
filename is the **season**, not the year the game was played):

```
pro-bowlers2010.csv ... pro-bowlers2025.csv
```

Each file is the *"Pro Bowl selections"* table for that season, saved with
PFR's **"Get table as CSV"** / **Share & Export → Get table as CSV** option, so
it keeps PFR's own header row. Only three columns are actually read:

| Column | Used for |
|--------|----------|
| `Player-additional` | PFR player id (e.g. `JackLa00`) — the primary key, crosswalked to an nflverse `gsis_id` via `nfl_data_py.import_ids()` |
| `Player` | fallback name match when the id crosswalk has no entry |
| `Pos` | narrows that fallback to the same position family |

The name fallback exists because the nflverse id map is fantasy-oriented and
carries almost no offensive linemen — without it roughly 19% of selections
(nearly all G/T/C/LS) were silently dropped. It only accepts a match when the
normalized name is **unique** within the position family and the player's career
spans the season; genuinely ambiguous names are left unmatched rather than
guessed. See `data.load_pro_bowlers`.

## The 2020 sidecar

PFR's selections table is missing for the 2020 season (the Pro Bowl itself was
cancelled). That season is supplied instead by `pro-bowlers2020-gsis.json`, a
plain `{"<season>": ["<gsis_id>", ...]}` map that is loaded only for seasons the
CSVs don't cover. Any season can be supplied this way if you'd rather not keep a
CSV for it.

## Running without it

If this directory holds no exports, `data.load_pro_bowlers` logs a line saying so
and returns empty. Every `pb_*` feature is then simply zero, the Pro Bowl absence
correction becomes a no-op, and the pipeline runs end to end and trains normally —
it just loses that one signal. Feature *count* and frame shape are unchanged.

For what that signal is worth, see "Pro Bowl absence correction" in the top-level
README: roughly −0.19 RMSE on team-games where two or more Pro Bowlers are ruled
out, and it helped in all five walk-forward test seasons.
