"""Build the shippable NFL Prediction Model web app.

Thin presentation layer: it asks the backend (:func:`nfl_margin_model.predict.
predict_games`) for per-game predictions, adds team visuals (logos/colours),
serialises to JSON, and bakes it into a single self-contained ``index.html``
(plus a ``preds.json`` for the hosted refresh mode).

Usage
-----
    # live data (refresh with the latest games):
    python -m nfl_margin_model.frontend.generate

    # fast rebuild from cached parquet (dev):
    python -m nfl_margin_model.frontend.generate --cache /path/to/parquet
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Allow running as a plain script (add project root so the package imports).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from nfl_margin_model import predict  # noqa: E402

HERE = os.path.dirname(__file__)
LOGO_PX = 64
_LOGO_CACHE = os.path.join(HERE, "logos_cache.json")
_FONT_CACHE = os.path.join(HERE, "fonts_cache.json")

# The design's two typefaces. Both are served by Google as single variable-weight
# woff2 files, so each is embedded once and covers every weight the page uses.
_FONT_CSS_URL = (
    "https://fonts.googleapis.com/css2"
    "?family=Archivo:wght@400;500;600;700;800"
    "&family=JetBrains+Mono:wght@400;500;700&display=swap"
)
_FONT_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")


def _load_fonts():
    """@font-face CSS with the woff2 files inlined as data URIs, cached to disk.

    The page must render identically with no network at all (it gets emailed
    around), so the fonts are embedded rather than linked. Only the ``latin``
    subset is kept -- that is what the UI text needs, and it holds the total to
    ~64KB of font data. A fetch failure degrades to the stylesheet's own
    system-font fallbacks rather than breaking the build.
    """
    import base64
    import re
    import ssl
    import urllib.request

    if os.path.exists(_FONT_CACHE):
        try:
            return json.load(open(_FONT_CACHE))["css"]
        except Exception:
            pass

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    def _get(url):
        req = urllib.request.Request(url, headers={"User-Agent": _FONT_UA})
        return urllib.request.urlopen(req, timeout=30, context=ctx).read()

    try:
        css = _get(_FONT_CSS_URL).decode()
    except Exception as e:
        print(f"  fonts: fetch failed ({e}); falling back to system fonts")
        return ""

    blocks = [b for b in re.findall(r"@font-face\s*\{[^}]*\}", css)
              if "U+0000-00FF" in b]                      # latin subset only
    families, out = {}, []
    for block in blocks:
        fam = re.search(r"font-family:\s*'([^']+)'", block)
        url = re.search(r"url\((https://[^)]+)\)", block)
        if not fam or not url or url.group(1) in families:
            continue                                       # variable font: once is enough
        try:
            raw = _get(url.group(1))
        except Exception as e:
            print(f"  fonts: {fam.group(1)}: {e}")
            continue
        families[url.group(1)] = fam.group(1)
        b64 = base64.b64encode(raw).decode()
        out.append(
            "@font-face{font-family:'%s';font-style:normal;font-weight:100 900;"
            "font-display:swap;src:url(data:font/woff2;base64,%s) format('woff2');}"
            % (fam.group(1), b64)
        )
    fonts_css = "".join(out)
    if fonts_css:
        json.dump({"css": fonts_css}, open(_FONT_CACHE, "w"))
        print(f"  fonts: embedded {len(families)} families "
              f"({len(fonts_css) // 1024} KB base64)")
    return fonts_css


def _load_logos(abbrs, td):
    """abbr -> tiny base64 PNG data URI (downscaled), cached to disk.

    Logos are embedded so the page works offline and inside a strict-CSP host.
    First run fetches + downscales; later runs read the cache.
    """
    import base64
    import io
    import ssl
    import urllib.request

    cache = {}
    if os.path.exists(_LOGO_CACHE):
        try:
            cache = json.load(open(_LOGO_CACHE))
        except Exception:
            cache = {}
    missing = [a for a in abbrs if a not in cache]
    if missing:
        from PIL import Image
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        for a in missing:
            try:
                url = td.loc[a, "team_logo_espn"] if a in td.index else None
                raw = urllib.request.urlopen(url, timeout=20, context=ctx).read()
                im = Image.open(io.BytesIO(raw)).convert("RGBA")
                im.thumbnail((LOGO_PX, LOGO_PX), Image.LANCZOS)
                buf = io.BytesIO()
                im.save(buf, format="PNG", optimize=True)
                cache[a] = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
            except Exception as e:
                print(f"  logo {a}: {e}")
                cache[a] = None
        json.dump(cache, open(_LOGO_CACHE, "w"))
    return cache


def _team_meta(abbrs, with_logos=True):
    """abbr -> {name, color, conf, div, logo} for display.

    With ``with_logos=False`` no logo is fetched or embedded and every team's
    ``logo`` is ``None``. The template already renders that case: a team-colour
    tile carrying the abbreviation, with the logo image overlaid only when one
    is present. That build carries no NFL club marks and is therefore safe to
    publish; it also needs neither Pillow nor network access for logos.
    """
    from nfl_margin_model import fetch
    td = fetch.team_desc().set_index("team_abbr")
    logos = _load_logos(abbrs, td) if with_logos else {}

    def meta(ab):
        base = dict(abbr=ab, name=ab, color="#33507f", conf="", div="")
        if ab in td.index:
            r = td.loc[ab]
            base.update(name=r["team_name"], color=r["team_color"],
                        conf=r["team_conf"], div=r["team_division"])
        base["logo"] = logos.get(ab)
        return base

    return {a: meta(a) for a in abbrs}


def build_payload(cache=None, generated="today", train_through=predict.TRAIN_THROUGH,
                  with_logos=True):
    """Assemble the web payload: backend predictions + team visuals."""
    records, meta = predict.predict_games(cache=cache, generated=generated,
                                          train_through=train_through)
    teams = sorted({g["home"] for g in records} | {g["away"] for g in records})
    return dict(meta=meta, teams=_team_meta(teams, with_logos=with_logos),
                games=records)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None, help="dir of cached parquet (dev)")
    ap.add_argument("--generated", default=None, help="date stamp for the footer")
    ap.add_argument("--train-through", type=int, default=predict.TRAIN_THROUGH,
                    help="last season used for training; later seasons are shown")
    ap.add_argument("--no-logos", action="store_true",
                    help="build without NFL club logos (team-colour chips with "
                         "abbreviations instead). This is the variant that is "
                         "tracked in git and safe to publish.")
    ap.add_argument("--out", default=None,
                    help="output path (default: index.html, or "
                         "index_no_logos.html with --no-logos)")
    args = ap.parse_args()

    from datetime import date
    stamp = args.generated or date.today().isoformat()

    stem = "index_no_logos" if args.no_logos else "index"
    out_path = args.out or os.path.join(HERE, f"{stem}.html")
    preds_path = os.path.join(
        HERE, "preds_no_logos.json" if args.no_logos else "preds.json")

    payload = build_payload(cache=args.cache, generated=stamp,
                            train_through=args.train_through,
                            with_logos=not args.no_logos)
    data_json = json.dumps(payload, separators=(",", ":"))

    with open(os.path.join(HERE, "template.html"), encoding="utf-8") as f:
        html = f.read()
    html = html.replace("/*__DATA__*/", data_json)
    html = html.replace("/*__FONTS__*/", _load_fonts())
    # Point the page's optional refresh-fetch at its own payload, so a hosted
    # --no-logos build never reloads the logo-bearing one.
    html = html.replace("/*__PREDS__*/", os.path.basename(preds_path))

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    with open(preds_path, "w", encoding="utf-8") as f:
        f.write(data_json)

    print(f"wrote {out_path}  ({len(payload['games'])} games, "
          f"{len(payload['teams'])} teams, {len(html)//1024} KB)")
    print(f"trained through {payload['meta']['train_through']}, "
          f"showing seasons {payload['meta']['display_seasons']}")


if __name__ == "__main__":
    main()
