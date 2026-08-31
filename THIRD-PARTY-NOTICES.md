# Third-party notices

The MIT license in [LICENSE](LICENSE) covers **the source code in this
repository**. It does not, and cannot, grant rights to third-party material that
the build pulls in. This file records what that material is and on what terms it
arrives.

## Fonts — bundled, and covered here

`python -m nfl_margin_model.frontend.generate` downloads two typefaces from
Google Fonts and embeds them in the page it builds. Both are licensed under the
**SIL Open Font License, Version 1.1**, whose full text is included at
[`licenses/OFL-1.1.txt`](licenses/OFL-1.1.txt). The OFL permits exactly this kind
of bundling and redistribution, provided the copyright notice and the license
accompany the font — which is what this file and that license file are for.

The notices below are reproduced from the `name` table of the font binaries
themselves, not transcribed from a web page:

```
Copyright 2020 The Archivo Project Authors
(https://github.com/Omnibus-Type/Archivo)

Copyright 2020 The JetBrains Mono Project Authors
(https://github.com/JetBrains/JetBrainsMono)
```

Neither font is sold, and neither is distributed under a Reserved Font Name that
this project modifies.

`nfl_margin_model/frontend/fonts_cache.json` is the only tracked file containing
font data.

## NFL club logos — deliberately NOT redistributed

The generated page displays each team's logo, fetched at build time from ESPN via
nflverse's `team_logo_espn` field and cached locally in
`nfl_margin_model/frontend/logos_cache.json`.

**That cache, and every built page that embeds it, is untracked on purpose.**
NFL club marks and the artwork expressing them belong to the National Football
League and its member clubs. They are not this project's to license, and no
notice or disclaimer would change that — so rather than redistribute them under
an MIT banner that cannot cover them, this repository simply does not carry them.
Each local build fetches its own copies, which is a use for identification only
and implies no endorsement by or affiliation with the NFL or any club.

So the frontend ships in two variants:

| Build | Command | Tracked? |
|---|---|---|
| **No logos** — team-colour tiles with abbreviations | `generate --no-logos` | **yes**, safe to publish |
| **With logos** — club marks embedded | `generate` | no, local only |

They are the same page: `teamTile` in `template.html` always draws the colour
tile and abbreviation, and overlays the logo image only when one is present, so
the published variant is the design minus the marks rather than a fallback.

Tracked and publishable:

```
nfl_margin_model/frontend/index_no_logos.html
nfl_margin_model/frontend/preds_no_logos.json
```

Generated locally and **not** tracked:

```
nfl_margin_model/frontend/index.html
nfl_margin_model/frontend/index_offline.html
nfl_margin_model/frontend/logos_cache.json
nfl_margin_model/frontend/preds.json
```

## Data

Game data comes from [nflverse](https://github.com/nflverse) at runtime via
`nfl_data_py` and is not redistributed here. Pro Bowl selections come from manual
Pro-Football-Reference exports, which are likewise untracked — see
[`nfl_margin_model/pro-bowl/README.md`](nfl_margin_model/pro-bowl/README.md).

## Dependencies

Installed from PyPI, not vendored: `nfl_data_py` (MIT), `pandas`, `numpy` and
`scikit-learn` (BSD 3-Clause), `xgboost` and `pyarrow` (Apache 2.0), `Pillow`
(MIT-CMU). All are permissive and compatible with this project's MIT license.
