"""Build the fully static, script-free edition of the NFL prediction app.

Why this exists
---------------
``index.html`` draws every game card in JavaScript. iMessage, iOS Mail and most
in-app attachment previews render HTML in a restricted WebKit that does not run
scripts, so the shell appears with an empty middle. This writes a second file
that needs no JavaScript at all: the cards are baked into the markup, week
switching / filtering / sorting run on hidden radio inputs plus CSS sibling
selectors, and each game expands through a native ``<details>`` element.

Everything (predictions, logos, fonts) is embedded, so the output is one
self-contained file that works with no network and no scripting.

Usage
-----
    # after refreshing predictions:
    python -m nfl_margin_model.frontend.generate
    python -m nfl_margin_model.frontend.render_static

    # or run it directly:
    python nfl_margin_model/frontend/render_static.py

Reads ``preds.json`` (+ ``fonts_cache.json`` when present) from its own
directory and writes ``index_offline.html`` beside them.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from html import escape as esc

HERE = os.path.dirname(os.path.abspath(__file__))
ACCENT = "#f7c948"
MONO = "'JetBrains Mono', monospace"
MINUS = "\u2212"        # true minus sign, matches the design
DASH = "\u2014"         # em dash, used for "no value"
MID = "\u00b7"          # middot
NDASH = "\u2013"


# --------------------------------------------------------------------------
# small helpers (ports of the ones in template.html)
# --------------------------------------------------------------------------
def vivid(hex_color):
    """Lift a team colour until it separates from the #0a1526 bar track.

    Ten teams' primaries (CHI #0B162A, HOU, PIT, LV, DAL, DEN, NE, SEA, GB,
    NYJ) are as dark as the track, so the raw hex renders the win-probability
    segment as empty track. Mixing toward white keeps the hue recognisable.
    """
    m = re.fullmatch(r"#?([0-9a-fA-F]{6})", str(hex_color or ""))
    if not m:
        return "#5b7298"
    n = int(m.group(1), 16)
    r, g, b = (n >> 16) & 255, (n >> 8) & 255, n & 255
    for _ in range(12):
        if (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255 >= 0.22:
            break
        r += (255 - r) * 0.12
        g += (255 - g) * 0.12
        b += (255 - b) * 0.12
    return "#%02x%02x%02x" % (round(r), round(g), round(b))


def tone(edge):
    """Edge tier colour + label, matching the sidebar key."""
    if edge is None:
        return "#445a7e", "NO LINE"
    e = abs(edge)
    if e >= 3:
        return "#4fd1a5", "STRONG"
    if e >= 1.5:
        return "#f0b429", "LEAN"
    return "#445a7e", "ALIGNED"


def side(margin, home, away):
    """Which team a signed home-margin favours, and by how much."""
    return (home if margin >= 0 else away), "%.1f" % abs(margin)


def moneyline(p):
    """Win probability -> American odds (the backend has no moneylines)."""
    if p is None or not (0 < p < 1):
        return DASH
    v = -round(100 * p / (1 - p)) if p >= 0.5 else round(100 * (1 - p) / p)
    return ("+" if v > 0 else MINUS) + str(abs(v))


def bar_pos(v):
    """Signed margin -> percent along the -14 .. +14 projected-margin scale."""
    return max(3.0, min(97.0, ((v + 14) / 28) * 100))


def driver_label(k):
    return re.sub(r"\((\d+)g\)", r"(last \1 games)", str(k))


def signed(v, digits=2):
    if v is None:
        return DASH
    if abs(v) < 0.005:
        return "0.00"
    return ("+" if v > 0 else MINUS) + ("%.*f" % (digits, abs(v)))


# --------------------------------------------------------------------------
# stylesheet
# --------------------------------------------------------------------------
BASE_CSS = """
html,body{margin:0;padding:0;background:#070f1c}
*{box-sizing:border-box}
a{color:#f0b429;text-decoration:none}a:hover{color:#ffd166}
.sw{position:absolute;width:1px;height:1px;opacity:0;margin:0;pointer-events:none}
summary{display:block;list-style:none;cursor:pointer}
summary::-webkit-details-marker{display:none}
summary::marker{content:''}
.card{border-radius:14px;border:1px solid #16233a;background:linear-gradient(180deg,#0c1728 0%,#0a1425 100%);overflow:hidden;transition:border-color 160ms ease}
.card:hover{border-color:#2b4067}
details[open].card{border-color:#2b4067;box-shadow:0 12px 34px rgba(0,0,0,.4)}
details[open] .chev{transform:rotate(180deg);background:#1b3557;color:__ACCENT__}
.wkgroup{display:none}
.wkhead{display:none}
.emptynote{display:none;padding:60px;text-align:center;color:#5b7298;font-size:13px;border:1px dashed #1a2b46;border-radius:12px}
.rail-lbl:hover{background:#142440;color:#dfe8f6}
.chipl:hover{color:#cfe0f5}
::-webkit-scrollbar{width:10px;height:10px}
::-webkit-scrollbar-track{background:#070f1c}
::-webkit-scrollbar-thumb{background:#1c2c47;border-radius:6px;border:3px solid #070f1c}
.card{order:var(--ok)}
#s-edge:checked ~ .shell .card{order:var(--oe)}
#s-margin:checked ~ .shell .card{order:var(--om)}
#f-strong:checked ~ .shell .card[data-strong="0"]{display:none}
#f-dis:checked ~ .shell .card[data-dis="0"]{display:none}
@media (max-width:900px){
html,body{height:auto}
.shell{display:block;height:auto;overflow:visible}
.rail{width:100%;flex:none;height:auto;overflow:visible;border-right:0;border-bottom:1px solid #16233a}
.mainpane{overflow:visible}
.weeklist{flex-direction:row;flex-wrap:wrap;flex:none}
.rail-lbl{width:auto}
.hdrgrid{grid-template-columns:1fr;gap:14px}
.homeside{flex-direction:row;justify-content:flex-start}
.homeside .side{align-items:flex-start}
.expand{grid-template-columns:1fr}
.subrow{flex-wrap:wrap}
.statrow{flex-wrap:wrap}
}
"""


def build_css(max_week, empty_weeks=None):
    """Per-control rules: the inputs are visually hidden, so every bit of state
    (active week, active chip, focus ring, empty state) has to be projected onto
    the labels and panels through :checked / :focus-visible sibling selectors."""
    empty_weeks = empty_weeks or {}
    css = BASE_CSS.replace("__ACCENT__", ACCENT)
    for w in range(1, max_week + 1):
        css += (
            '#w-%(w)d:checked ~ .shell .wkgroup[data-week="%(w)d"]{display:flex}\n'
            '#w-%(w)d:checked ~ .shell .wkhead[data-week="%(w)d"]{display:flex}\n'
            '#w-%(w)d:checked ~ .shell label[for="w-%(w)d"]{background:#132540;color:#fff}\n'
            '#w-%(w)d:checked ~ .shell label[for="w-%(w)d"] .rail-bar{background:%(a)s}\n'
            '#w-%(w)d:checked ~ .shell label[for="w-%(w)d"] .rail-n{color:#93a7c4}\n'
            '#w-%(w)d:focus-visible ~ .shell label[for="w-%(w)d"]'
            "{outline:2px solid %(a)s;outline-offset:2px}\n"
            % {"w": w, "a": ACCENT}
        )
    for cid in ("f-all", "f-strong", "f-dis", "s-kick", "s-edge", "s-margin"):
        css += ('#%(c)s:checked ~ .shell label[for="%(c)s"]{background:#1b3557;color:#fff}\n'
                '#%(c)s:focus-visible ~ .shell label[for="%(c)s"]'
                "{outline:2px solid %(a)s;outline-offset:2px}\n"
                % {"c": cid, "a": ACCENT})
    # Reveal the "nothing matches" note only for the week/filter pairs that
    # really are empty, so no :has() is needed (old WebKit has to render this).
    for cid, weeks in empty_weeks.items():
        for w in weeks:
            css += ('#%s:checked ~ .shell .empty-%s[data-week="%d"]{display:block}\n'
                    % (cid, cid, w))
    return css


# --------------------------------------------------------------------------
# markup pieces
# --------------------------------------------------------------------------
def chip(cid, label):
    return ('<label class="chipl" for="%s" style="padding:6px 13px;border-radius:7px;'
            "cursor:pointer;font-family:Archivo,sans-serif;font-size:11.5px;"
            'font-weight:600;color:#7089ac;white-space:nowrap">%s</label>'
            % (cid, esc(label)))


def fav_pill(on):
    if not on:
        return "display:none"
    return ("padding:2px 7px;border-radius:4px;font-family:%s;font-size:9px;"
            "font-weight:700;letter-spacing:.06em;background:%s22;color:%s;"
            "border:1px solid %s55" % (MONO, ACCENT, ACCENT, ACCENT))


def tile(abbr, teams):
    t = teams.get(abbr) or {}
    c = vivid(t.get("color") or "#33507f")
    logo = t.get("logo")
    out = ('<div style="width:44px;height:44px;flex:0 0 44px;border-radius:10px;'
           "display:grid;place-items:center;background:%s1f;border:1px solid %s66;"
           'position:relative;overflow:hidden">' % (c, c))
    out += ('<span style="font-family:%s;font-size:12px;font-weight:700;'
            'color:#93a7c4">%s</span>' % (MONO, esc(abbr)))
    if logo:
        out += ('<img src="%s" alt="" style="position:absolute;inset:4px;'
                "width:calc(100%% - 8px);height:calc(100%% - 8px);"
                'object-fit:contain">' % logo)
    return out + "</div>"


def cell(label, value, colour, small):
    size = "11.5" if small else "12.5"
    weight = "" if small else "font-weight:700;"
    return ('<div style="display:flex;align-items:flex-start;justify-content:space-between;'
            'gap:12px;padding:10px 13px;background:#0c1728">'
            '<span style="font-size:11.5px;color:#93a7c4;flex:0 0 auto">%s</span>'
            '<span style="font-family:%s;font-size:%spx;%scolor:%s;text-align:right">'
            "%s</span></div>"
            % (esc(label), MONO, size, weight, colour, esc(value)))


def driver_rows(game):
    drivers = game.get("drivers") or {}
    if not drivers:
        return ""
    values = [abs(v) for v in drivers.values() if v is not None]
    scale = max([0.01] + values)
    rows = ""
    for key, v in drivers.items():
        dead = v is None or abs(v) < 0.005
        colour = "#5b7298" if dead else ("#4fd1a5" if v > 0 else "#e8734a")
        mag = 0.0 if dead else min(1.0, abs(v) / scale) * 50
        bar = ""
        if not dead:
            bar = ('<div style="position:absolute;top:0;bottom:0;border-radius:3px;'
                   'background:%s;%swidth:%.1f%%"></div>'
                   % (colour, "left:50%;" if v > 0 else "right:50%;", mag))
        rows += (
            '<div style="display:flex;align-items:center;gap:10px;padding:9px 13px;background:#0c1728">'
            '<span style="font-size:11.5px;color:#93a7c4;flex:1;min-width:0">%s</span>'
            '<div style="position:relative;width:62px;height:5px;border-radius:3px;'
            'background:#14243d;flex:0 0 62px">'
            '<div style="position:absolute;top:0;bottom:0;left:50%%;width:1px;background:#2b4067"></div>'
            "%s</div>"
            '<span style="font-family:%s;font-size:11.5px;font-weight:700;color:%s;'
            'width:54px;text-align:right">%s</span></div>'
            % (esc(driver_label(key)), bar, MONO, colour, esc(signed(v)))
        )
    return (
        '<div style="font-family:%s;font-size:9px;letter-spacing:.14em;color:#62789a;'
        'margin-top:2px">MODEL DRIVERS</div>'
        '<div style="display:flex;flex-direction:column;gap:1px;border-radius:9px;'
        'overflow:hidden;border:1px solid #17263e">%s</div>' % (MONO, rows)
    )


# --------------------------------------------------------------------------
# per-game numbers
# --------------------------------------------------------------------------
def prep(g):
    has_mkt = g.get("market_spread") is not None
    hp = 0.5 if g.get("win_prob_home") is None else g["win_prob_home"]
    home_pct = round(hp * 100)
    mwp = g.get("market_win_prob_home")
    mwp_pct = None if mwp is None else round(mwp * 100)
    edge = g.get("spread_edge")
    m_team, m_txt = side(g["pred_margin"], g["home"], g["away"])
    k_team, k_txt = (side(g["market_spread"], g["home"], g["away"])
                     if has_mkt else (None, None))
    return {
        "g": g, "has_mkt": has_mkt, "edge": edge,
        "hp": hp, "ap": 1 - hp, "home_pct": home_pct, "away_pct": 100 - home_pct,
        "mwp_pct": mwp_pct,
        # gap is the difference of the two DISPLAYED integers, so the panel can
        # never contradict the two percentages sitting above it
        "gap": None if mwp_pct is None else home_pct - mwp_pct,
        "m_team": m_team, "m_txt": m_txt, "k_team": k_team, "k_txt": k_txt,
        "edge_team": None if edge is None else (g["home"] if edge > 0 else g["away"]),
        "strong": edge is not None and abs(edge) >= 3,
        "disagrees": has_mkt and (g["pred_margin"] >= 0) != (g["market_spread"] >= 0),
        "played": g.get("actual_margin") is not None,
    }


def card(r, orders, teams):
    g = r["g"]
    tc, tlabel = tone(r["edge"])
    a_col = vivid((teams.get(g["away"]) or {}).get("color"))
    h_col = vivid((teams.get(g["home"]) or {}).get("color"))
    a_name = (teams.get(g["away"]) or {}).get("name", g["away"])
    h_name = (teams.get(g["home"]) or {}).get("name", g["home"])
    model_label = "%s %s%s" % (r["m_team"], MINUS, r["m_txt"])
    market_label = ("%s %s%s" % (r["k_team"], MINUS, r["k_txt"])
                    if r["has_mkt"] else "no line")

    if r["edge"] is None:
        diff = DASH
        edge_text = "market pending"
    else:
        diff = "%.1f pts \u2192 %s" % (abs(r["edge"]), r["edge_team"])
        edge_text = "%.1f pts %s" % (abs(r["edge"]), r["edge_team"])

    if r["gap"] is None:
        gap_text = DASH
    elif r["gap"] == 0:
        gap_text = "0 pts"
    else:
        gap_text = ("+" if r["gap"] > 0 else MINUS) + "%d pts" % abs(r["gap"])

    cmp_rows = [
        ("Model spread", model_label, ACCENT),
        ("Market spread", market_label, "#dfe8f6"),
        ("Difference", diff, tc),
        ("Model win prob %s %s" % (MID, g["home"]), "%d%%" % r["home_pct"], "#dfe8f6"),
        ("Market implied %s %s" % (MID, g["home"]),
         DASH if r["mwp_pct"] is None else "%d%%" % r["mwp_pct"], "#93a7c4"),
        ("Probability gap", gap_text,
         tc if (r["gap"] is not None and abs(r["gap"]) >= 5) else "#93a7c4"),
    ]
    result = (("%s %s %s %s %s" % (g["away"], g.get("away_score"), NDASH,
                                   g.get("home_score"), g["home"]))
              if r["played"] else "not played")
    info_rows = [
        ("Kickoff", g.get("kickoff") or DASH),
        ("Venue", g.get("venue") or DASH),
        ("Site", "Neutral" if g.get("neutral_site") else "%s home" % g["home"]),
        ("Week", ("Postseason %s " % MID if g.get("is_playoff") else "") + "Week %d" % g["week"]),
        ("Moneyline", "%s %s / %s %s" % (g["away"], moneyline(r["ap"]),
                                         g["home"], moneyline(r["hp"]))),
        ("Result", result),
    ]

    out = ('<details class="card" data-strong="%d" data-dis="%d" '
           'style="--ok:%d;--oe:%d;--om:%d"><summary>'
           % (1 if r["strong"] else 0, 1 if r["disagrees"] else 0,
              orders[0], orders[1], orders[2]))

    # --- headline row -----------------------------------------------------
    out += ('<div class="hdrgrid" style="display:grid;grid-template-columns:'
            "minmax(0,1fr) minmax(240px,1.45fr) minmax(0,1fr);align-items:center;"
            'gap:20px;padding:18px 20px">')

    out += '<div style="display:flex;align-items:center;gap:13px;min-width:0">' + tile(g["away"], teams)
    out += '<div style="display:flex;flex-direction:column;gap:3px;min-width:0;flex:1">'
    out += ('<div style="display:flex;align-items:center;gap:7px">'
            '<span style="font-family:%s;font-size:14.5px;font-weight:700;letter-spacing:.02em">%s</span>'
            '<span style="%s">FAV</span></div>'
            % (MONO, esc(g["away"]), fav_pill(r["m_team"] == g["away"])))
    out += ('<div style="font-size:12px;color:#8fa3c1;white-space:nowrap;overflow:hidden;'
            'text-overflow:ellipsis">%s</div>' % esc(a_name))
    out += ('<div style="font-family:%s;font-size:10.5px;color:#5b7298">ML %s</div></div></div>'
            % (MONO, esc(moneyline(r["ap"]))))

    out += '<div style="display:flex;flex-direction:column;gap:8px">'
    out += ('<div style="display:flex;align-items:baseline;justify-content:center;gap:9px">'
            '<span style="font-family:%s;font-size:9px;letter-spacing:.14em;color:#62789a">MODEL</span>'
            '<span style="font-family:%s;font-size:21px;font-weight:700;color:%s;'
            'letter-spacing:-.01em">%s</span></div>'
            % (MONO, MONO, ACCENT, esc(model_label)))
    out += ('<div style="position:relative;height:12px;border-radius:6px;overflow:hidden;'
            'background:#0a1526;display:flex;box-shadow:inset 0 1px 3px rgba(0,0,0,.5)">'
            '<div style="width:%d%%;background:%s"></div>'
            '<div style="flex:1;background:%s"></div>'
            '<div style="position:absolute;top:-2px;bottom:-2px;left:%d%%;width:3px;'
            "margin-left:-1.5px;border-radius:2px;background:#f7f3e8;"
            'box-shadow:0 0 8px rgba(247,243,232,.55)"></div>'
            '<div style="position:absolute;top:0;bottom:0;left:50%%;width:1px;'
            'background:rgba(255,255,255,.28)"></div></div>'
            % (r["away_pct"], a_col, h_col, r["away_pct"]))
    out += ('<div style="display:flex;justify-content:space-between;font-family:%s;'
            'font-size:11px;color:#8fa3c1">'
            '<span><b style="color:#dfe8f6">%d%%</b> %s</span>'
            '<span style="color:#62789a;letter-spacing:.1em;font-size:9px;align-self:center">'
            "WIN PROBABILITY</span>"
            '<span>%s <b style="color:#dfe8f6">%d%%</b></span></div></div>'
            % (MONO, r["away_pct"], esc(g["away"]), esc(g["home"]), r["home_pct"]))

    out += ('<div class="homeside" style="display:flex;align-items:center;gap:13px;'
            'justify-content:flex-end;min-width:0">'
            '<div class="side" style="display:flex;flex-direction:column;gap:3px;'
            'align-items:flex-end;min-width:0;flex:1">')
    out += ('<div style="display:flex;align-items:center;gap:7px"><span style="%s">FAV</span>'
            '<span style="font-family:%s;font-size:14.5px;font-weight:700;letter-spacing:.02em">'
            "%s</span></div>"
            % (fav_pill(r["m_team"] == g["home"]), MONO, esc(g["home"])))
    out += ('<div style="font-size:12px;color:#8fa3c1;white-space:nowrap;overflow:hidden;'
            'text-overflow:ellipsis">%s</div>' % esc(h_name))
    out += ('<div style="font-family:%s;font-size:10.5px;color:#5b7298">ML %s</div></div>%s</div></div>'
            % (MONO, esc(moneyline(r["hp"])), tile(g["home"], teams)))

    # --- edge / market strip ---------------------------------------------
    out += '<div class="subrow" style="display:flex;align-items:center;gap:10px;padding:0 20px 15px">'
    out += ('<div style="display:flex;align-items:center;gap:8px;padding:5px 11px;'
            'border-radius:20px;font-family:%s;color:%s;background:%s14;border:1px solid %s40">'
            '<span style="width:6px;height:6px;border-radius:50%%;background:%s"></span>'
            '<span style="letter-spacing:.1em;font-size:9px">%s</span>'
            '<span style="font-weight:700;font-size:11.5px">%s</span></div>'
            % (MONO, tc, tc, tc, tc, tlabel, esc(edge_text)))
    out += ('<div style="font-family:%s;color:#7089ac;display:flex;align-items:baseline;gap:8px">'
            '<span style="color:#62789a;letter-spacing:.1em;font-size:9.5px">MARKET</span>'
            '<span style="color:#d6e2f2;font-weight:700;font-size:15px;letter-spacing:-.01em">'
            "%s</span></div>" % (MONO, esc(market_label)))
    if g.get("neutral_site"):
        out += ('<div style="display:flex;align-items:center;gap:6px;padding:4px 9px;'
                'border-radius:20px;border:1px solid #2b4067;font-family:%s;font-size:9px;'
                'letter-spacing:.1em;color:#8fa3c1"><span style="width:5px;height:5px;'
                'background:#f0b429;transform:rotate(45deg)"></span> NEUTRAL SITE</div>' % MONO)
    out += '<div style="flex:1"></div>'
    out += ('<div style="font-family:%s;font-size:10.5px;color:#62789a">%s</div>'
            % (MONO, esc(g.get("kickoff") or "")))
    out += ('<div style="font-family:%s;font-size:9px;letter-spacing:.1em;color:%s;'
            'padding:4px 9px;border-radius:5px;background:#0d1a2e">%s</div>'
            % (MONO, "#93a7c4" if r["played"] else "#5b7298",
               "FINAL" if r["played"] else "PREDICTION ONLY"))
    out += ('<div class="chev" style="width:26px;height:26px;border-radius:7px;display:grid;'
            'place-items:center;background:#0d1a2e;color:#5b7298;transition:transform 180ms ease">'
            '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">'
            '<polyline points="6 9 12 15 18 9"></polyline></svg></div></div></summary>')

    # --- expanded panel ---------------------------------------------------
    out += ('<div class="expand" style="border-top:1px solid #17263e;background:#0a1425;'
            'padding:20px;display:grid;grid-template-columns:1.1fr 1fr .9fr;gap:18px">')
    out += ('<div style="display:flex;flex-direction:column;gap:12px">'
            '<div style="font-family:%s;font-size:9px;letter-spacing:.14em;color:#62789a">'
            "MODEL VS MARKET</div>"
            '<div style="display:flex;flex-direction:column;gap:1px;border-radius:9px;'
            'overflow:hidden;border:1px solid #17263e">%s</div></div>'
            % (MONO, "".join(cell(a, b, c, False) for a, b, c in cmp_rows)))

    mkt_marker = ""
    if r["has_mkt"]:
        mkt_marker = ('<div style="position:absolute;top:-8px;left:%.2f%%;'
                      "transform:translateX(-50%%);display:flex;flex-direction:column;"
                      'align-items:center"><div style="width:2px;height:24px;background:#7089ac">'
                      '</div><div style="font-family:%s;font-size:8.5px;color:#7089ac;'
                      'letter-spacing:.1em;margin-top:3px">MKT</div></div>'
                      % (bar_pos(g["market_spread"]), MONO))
    out += ('<div style="display:flex;flex-direction:column;gap:12px">'
            '<div style="font-family:%s;font-size:9px;letter-spacing:.14em;color:#62789a">'
            "PROJECTED MARGIN</div>"
            '<div style="padding:18px 14px 12px;border-radius:9px;background:#0c1728;'
            'border:1px solid #17263e;display:flex;flex-direction:column;gap:10px">'
            '<div style="position:relative;height:8px;border-radius:4px;'
            'background:linear-gradient(90deg,%s 0%%,#14243d 46%%,#14243d 54%%,%s 100%%)">'
            '<div style="position:absolute;top:50%%;left:50%%;width:1px;height:16px;'
            'margin-top:-8px;background:#2b4067"></div>%s'
            '<div style="position:absolute;top:-14px;left:%.2f%%;transform:translateX(-50%%);'
            'display:flex;flex-direction:column;align-items:center">'
            '<div style="width:3px;height:36px;border-radius:2px;background:%s;'
            'box-shadow:0 0 10px rgba(247,201,72,.5)"></div>'
            '<div style="font-family:%s;font-size:8.5px;color:%s;letter-spacing:.1em;'
            'margin-top:3px;font-weight:700">MODEL</div></div></div>'
            '<div style="display:flex;justify-content:space-between;font-family:%s;'
            'font-size:9.5px;margin-top:26px">'
            '<span style="color:#5b7298">%s +14</span>'
            '<span style="color:#445a7e">PICK</span>'
            '<span style="color:#5b7298">%s +14</span></div></div>%s</div>'
            % (MONO, a_col, h_col, mkt_marker, bar_pos(g["pred_margin"]), ACCENT,
               MONO, ACCENT, MONO, esc(g["away"]), esc(g["home"]), driver_rows(g)))

    out += ('<div style="display:flex;flex-direction:column;gap:12px">'
            '<div style="font-family:%s;font-size:9px;letter-spacing:.14em;color:#62789a">'
            "GAME INFO</div>"
            '<div style="display:flex;flex-direction:column;gap:1px;border-radius:9px;'
            'overflow:hidden;border:1px solid #17263e">%s</div></div>'
            % (MONO, "".join(cell(a, b, "#dfe8f6", True) for a, b in info_rows)))

    return out + "</div></details>"


# --------------------------------------------------------------------------
# page
# --------------------------------------------------------------------------
def build(payload, fonts_css=""):
    teams = payload.get("teams") or {}
    meta = payload.get("meta") or {}
    games = payload.get("games") or []

    season = meta.get("project_season") or (meta.get("display_seasons") or [None])[0]
    if season is None and games:
        season = games[-1]["season"]
    in_season = [g for g in games if g["season"] == season]

    counts = {}
    for g in in_season:
        counts[g["week"]] = counts.get(g["week"], 0) + 1
    live = sorted(counts)
    max_week = max([18] + live) if live else 18
    start = meta.get("projected_week")
    if not start or not counts.get(start):
        start = live[0] if live else 1

    radios = ""
    for w in range(1, max_week + 1):
        radios += ('<input class="sw" type="radio" name="wk" id="w-%d"%s%s>'
                   % (w, " checked" if w == start else "",
                      "" if counts.get(w) else " disabled"))
    radios += ('<input class="sw" type="radio" name="flt" id="f-all" checked>'
               '<input class="sw" type="radio" name="flt" id="f-strong">'
               '<input class="sw" type="radio" name="flt" id="f-dis">'
               '<input class="sw" type="radio" name="srt" id="s-kick" checked>'
               '<input class="sw" type="radio" name="srt" id="s-edge">'
               '<input class="sw" type="radio" name="srt" id="s-margin">')

    rail = ""
    for w in range(1, max_week + 1):
        n = counts.get(w, 0)
        rail += ('<label class="rail-lbl" for="w-%d" style="display:flex;align-items:center;'
                 "gap:10px;width:100%%;padding:9px 10px;border-radius:8px;"
                 'font-family:Archivo,sans-serif;cursor:%s;opacity:%s;color:#8fa3c1">'
                 '<span class="rail-bar" style="width:3px;height:16px;border-radius:2px;'
                 'background:transparent"></span>'
                 '<span style="flex:1;text-align:left;font-size:13px;font-weight:600">Week %d</span>'
                 '<span class="rail-n" style="font-family:%s;font-size:10.5px;color:#3b4f70">'
                 "%s</span></label>"
                 % (w, "pointer" if n else "default", "1" if n else ".4", w, MONO,
                    n if n else DASH))

    heads, groups = "", ""
    empty_weeks = {"f-strong": [], "f-dis": []}
    for w in live:
        wk = [prep(g) for g in in_season if g["week"] == w]
        if not any(r["strong"] for r in wk):
            empty_weeks["f-strong"].append(w)
        if not any(r["disagrees"] for r in wk):
            empty_weeks["f-dis"].append(w)
        with_edge = sorted([r for r in wk if r["edge"] is not None],
                           key=lambda r: -abs(r["edge"]))
        best = with_edge[0] if with_edge else None
        heads += ('<div class="wkhead" data-week="%d" style="align-items:flex-end;'
                  'justify-content:space-between;gap:24px;flex-wrap:wrap">'
                  '<div style="display:flex;flex-direction:column;gap:6px">'
                  '<div style="font-family:%s;font-size:9.5px;letter-spacing:.16em;'
                  'color:#62789a">%s REGULAR SEASON</div>'
                  '<div style="font-size:27px;font-weight:800;letter-spacing:-.025em">Week %d</div></div>'
                  '<div class="statrow" style="display:flex;gap:10px;padding-bottom:4px">'
                  '<div style="padding:9px 15px;border-radius:9px;background:#0d1a2e;'
                  "border:1px solid #1a2b46;display:flex;flex-direction:column;gap:3px;"
                  'min-width:92px"><div style="font-family:%s;font-size:8.5px;'
                  'letter-spacing:.12em;color:#62789a">GAMES</div>'
                  '<div style="font-family:%s;font-size:17px;font-weight:700">%d</div></div>'
                  '<div style="padding:9px 15px;border-radius:9px;background:#0d1a2e;'
                  "border:1px solid #1a2b46;display:flex;flex-direction:column;gap:3px;"
                  'min-width:118px"><div style="font-family:%s;font-size:8.5px;'
                  'letter-spacing:.12em;color:#62789a">BIGGEST EDGE</div>'
                  '<div style="font-family:%s;font-size:17px;font-weight:700;color:#4fd1a5">'
                  "%s</div></div></div></div>"
                  % (w, MONO, esc(str(season)), w, MONO, MONO, len(wk), MONO, MONO,
                     esc("%.1f %s" % (abs(best["edge"]), best["edge_team"])) if best else DASH))

        by_edge = sorted(wk, key=lambda r: -abs(r["edge"] or 0))
        by_margin = sorted(wk, key=lambda r: -abs(r["g"]["pred_margin"]))
        cards = ""
        for i, r in enumerate(wk):
            cards += card(r, (i + 1, by_edge.index(r) + 1, by_margin.index(r) + 1), teams)
        groups += ('<div class="wkgroup" data-week="%d" style="flex-direction:column;'
                   'gap:10px">%s</div>' % (w, cards))
        for cid in ("f-strong", "f-dis"):
            if w in empty_weeks[cid]:
                groups += ('<div class="emptynote empty-%s" data-week="%d">'
                           "No games match this filter.</div>" % (cid, w))

    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>NFL Prediction Model __MID__ __SEASON__ Week __START__</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>__FONTS__</style>
<style>__CSS__</style>
</head>
<body>
__RADIOS__
<div class="shell" style="display:flex;height:100vh;overflow:hidden;background:#070f1c;font-family:Archivo,system-ui,sans-serif;color:#e8eef8">

  <aside class="rail" style="width:268px;flex:0 0 268px;border-right:1px solid #16233a;background:linear-gradient(180deg,#0a1425 0%,#08111f 100%);display:flex;flex-direction:column;overflow:hidden">
    <div style="padding:22px 20px 18px;display:flex;gap:12px;align-items:flex-start">
      <div style="width:42px;height:42px;flex:0 0 42px;border-radius:10px;background:linear-gradient(145deg,#f7c948,#e0a318);display:grid;place-items:center;font-weight:800;font-size:13px;letter-spacing:.02em;color:#10192b;box-shadow:0 6px 18px rgba(224,163,24,.28)">NFL</div>
      <div style="display:flex;flex-direction:column;gap:5px;padding-top:2px">
        <div style="font-size:16px;font-weight:700;letter-spacing:-.01em;line-height:1.15">Prediction Model</div>
        <div style="font-family:__MONO__;font-size:9.5px;letter-spacing:.14em;color:#6d84a8;text-transform:uppercase">Point-margin engine</div>
      </div>
    </div>
    <div style="margin:0 20px 18px;padding:12px 14px;border-radius:10px;background:#0d1a2e;border:1px solid #182842;display:flex;align-items:center;justify-content:space-between;gap:10px">
      <div style="font-family:__MONO__;font-size:8.5px;letter-spacing:.13em;color:#62789a">SEASON</div>
      <div style="font-size:15px;font-weight:700;font-family:__MONO__">__SEASON__</div>
    </div>
    <div style="padding:0 20px 8px;display:flex;align-items:center;justify-content:space-between">
      <div style="font-family:__MONO__;font-size:9px;letter-spacing:.16em;color:#62789a">SCHEDULE</div>
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#445a7e" stroke-width="2"><rect x="3" y="5" width="18" height="16" rx="2"></rect><line x1="3" y1="10" x2="21" y2="10"></line><line x1="8" y1="3" x2="8" y2="7"></line><line x1="16" y1="3" x2="16" y2="7"></line></svg>
    </div>
    <div class="weeklist" style="flex:1;overflow-y:auto;padding:4px 14px 18px;display:flex;flex-direction:column;gap:2px">__RAIL__</div>
    <div style="padding:14px 20px 18px;border-top:1px solid #16233a;display:flex;flex-direction:column;gap:9px">
      <div style="font-family:__MONO__;font-size:8.5px;letter-spacing:.14em;color:#62789a">EDGE KEY</div>
      <div style="display:flex;flex-direction:column;gap:7px">
        <div style="display:flex;align-items:center;gap:8px;font-size:11px;color:#93a7c4"><span style="width:8px;height:8px;border-radius:2px;background:#4fd1a5"></span> Strong __MID__ 3.0+ pts</div>
        <div style="display:flex;align-items:center;gap:8px;font-size:11px;color:#93a7c4"><span style="width:8px;height:8px;border-radius:2px;background:#f0b429"></span> Lean __MID__ 1.5__NDASH__3.0</div>
        <div style="display:flex;align-items:center;gap:8px;font-size:11px;color:#93a7c4"><span style="width:8px;height:8px;border-radius:2px;background:#445a7e"></span> Aligned __MID__ under 1.5</div>
      </div>
    </div>
  </aside>

  <main class="mainpane" style="flex:1;overflow-y:auto;display:flex;flex-direction:column;min-width:0">
    <header style="position:sticky;top:0;z-index:20;background:rgba(8,17,31,.94);backdrop-filter:blur(12px);border-bottom:1px solid #16233a;padding:20px 30px 0">
      __HEADS__
      <div style="display:flex;align-items:center;gap:8px;padding:16px 0 14px;flex-wrap:wrap">
        <div style="display:flex;gap:4px;padding:3px;border-radius:9px;background:#0c1728;border:1px solid #1a2b46">__FILTERS__</div>
        <div style="width:1px;height:22px;background:#1a2b46;margin:0 4px"></div>
        <div style="display:flex;align-items:center;gap:8px;padding:0 4px">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#5b7298" stroke-width="2" stroke-linecap="round"><line x1="4" y1="7" x2="20" y2="7"></line><line x1="4" y1="12" x2="15" y2="12"></line><line x1="4" y1="17" x2="10" y2="17"></line></svg>
          <div style="display:flex;gap:4px">__SORTS__</div>
        </div>
      </div>
    </header>
    <div style="padding:8px 30px 40px">__GROUPS__</div>
    <footer style="padding:0 30px 40px;color:#445a7e;font-size:11.5px;line-height:1.7;font-family:__MONO__">
      Model: __MODEL__<br>Trained through __TRAIN__ __MID__ generated __GEN__ __MID__ market line: __SPREAD__
      <br>Out-of-sample point-margin predictions, graded against actual results __DASH__ not against the spread. For analysis, not wagering advice.
      <br>Offline edition __DASH__ no scripts, no network. Tap a game to expand it.
    </footer>
  </main>
</div>
</body>
</html>""" \
        .replace("__FONTS__", fonts_css) \
        .replace("__CSS__", build_css(max_week, empty_weeks)) \
        .replace("__RADIOS__", radios) \
        .replace("__RAIL__", rail) \
        .replace("__HEADS__", heads) \
        .replace("__GROUPS__", groups) \
        .replace("__FILTERS__", chip("f-all", "All") + chip("f-strong", "Strong edge")
                 + chip("f-dis", "Disagrees on winner")) \
        .replace("__SORTS__", chip("s-kick", "Kickoff") + chip("s-edge", "Edge")
                 + chip("s-margin", "Margin")) \
        .replace("__MODEL__", esc(str(meta.get("model") or ""))) \
        .replace("__TRAIN__", esc(str(meta.get("train_through") or ""))) \
        .replace("__GEN__", esc(str(meta.get("generated") or ""))) \
        .replace("__SPREAD__", esc(str(meta.get("spread_source") or "consensus"))) \
        .replace("__SEASON__", esc(str(season))) \
        .replace("__START__", str(start)) \
        .replace("__MONO__", MONO) \
        .replace("__MID__", MID) \
        .replace("__NDASH__", NDASH) \
        .replace("__DASH__", DASH)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds", default=os.path.join(HERE, "preds.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "index_offline.html"))
    args = ap.parse_args()

    with open(args.preds, encoding="utf-8") as f:
        payload = json.load(f)

    fonts = ""
    cache = os.path.join(HERE, "fonts_cache.json")
    if os.path.exists(cache):
        try:
            with open(cache, encoding="utf-8") as f:
                fonts = json.load(f).get("css", "")
        except Exception as e:                                  # noqa: BLE001
            print("  fonts: %s (falling back to system fonts)" % e)

    html = build(payload, fonts)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)

    print("wrote %s  (%d games, %d KB, no JavaScript)"
          % (args.out, len(payload.get("games") or []), len(html) // 1024))


if __name__ == "__main__":
    main()
