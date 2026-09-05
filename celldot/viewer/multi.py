"""One server, several CellDot results: the public demo.

    from celldot.viewer.multi import create_multi_app
    app = create_multi_app([
        dict(key="BreastCancer", name="Breast cancer", blurb="10x Xenium, FFPE section",
             bundle="/data/BreastCancer/viewer_bundle", view="/data/BreastCancer/celldot_view.json"),
        ...])                                   # then uvicorn.run(app, host="0.0.0.0", port=7860)

A landing page at / lists the datasets; each viewer lives at /d/<key>/ (the viewer page uses relative URLs, so it runs
unchanged under the prefix). View files are read-only here (no "set as default view" from the page).
"""
import html, os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from .server import create_app

PAGE = """<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
  :root {{ --bg:#0b0d12; --panel:#141821; --line:#262c38; --fg:#e8eaf0; --mut:#98a2b3; --acc:#4f8ef7; --keep:#8a8f98; --move:#2f8fe0; --drop:#e5484d; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 -apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }}
  main {{ max-width:880px; margin:0 auto; padding:48px 24px 64px; }}
  h1 {{ font-size:28px; margin:0 0 6px; letter-spacing:.01em; }}
  .lead {{ color:var(--mut); max-width:64ch; margin:0 0 28px; }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(260px, 1fr)); gap:14px; }}
  a.card {{ display:block; background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:16px 18px; color:inherit; text-decoration:none; }}
  a.card:hover {{ border-color:var(--acc); }}
  .card h2 {{ font-size:17px; margin:0 0 2px; }}
  .card .blurb {{ color:var(--mut); font-size:13px; min-height:2.6em; }}
  .nums {{ font-variant-numeric:tabular-nums; font-size:13px; margin:10px 0 8px; color:var(--mut); }}
  .nums b {{ color:var(--fg); font-weight:600; }}
  .fate {{ display:flex; height:6px; border-radius:3px; overflow:hidden; background:#222838; }}
  .fate span {{ display:block; height:100%; }}
  .legend {{ display:flex; gap:12px; font-size:12px; color:var(--mut); margin-top:6px; }} .legend span {{ white-space:nowrap; }}
  .sw {{ width:9px; height:9px; border-radius:2px; display:inline-block; margin-right:4px; vertical-align:middle; }}
  .open {{ margin-top:10px; font-size:13px; color:var(--acc); }}
  .how {{ margin-top:30px; border-top:1px solid var(--line); padding-top:18px; color:var(--mut); font-size:13.5px; max-width:70ch; }}
  .how b {{ color:var(--fg); font-weight:600; }}
  .kbd {{ font-family:ui-monospace,Menlo,monospace; background:#222838; border:1px solid var(--line); border-radius:4px; padding:0 5px; font-size:12px; color:var(--fg); }}
  .links {{ margin-top:14px; font-size:13px; }} .links a {{ color:var(--acc); text-decoration:none; margin-right:16px; }}
</style>
<main>
  <h1>{title}</h1>
  <p class="lead">{intro}</p>
  <div class="cards">{cards}</div>
  <div class="how">
    <b>How to read a viewer.</b> Cells are drawn as polygons, coloured by cell type. Molecules of the chosen genes are drawn as dots:
    grey ones stay in their cell, blue ones were reassigned to a neighbouring cell (an arrow points to it) and red ones were removed
    as ambient background. Click a cell to list every one of its molecules with its fate. Keys: <span class="kbd">F</span> fit,
    <span class="kbd">Esc</span> clear, <span class="kbd">A</span> arrows, <span class="kbd">S</span> same-type moves,
    <span class="kbd">P</span> save PNG.
    <div class="links">{links}</div>
  </div>
</main>
"""


def _redirect(path):
    async def go(): return RedirectResponse(path)
    return go


def create_multi_app(datasets, title="CellDot viewer", intro=None, threads=2, memory_limit="3GB", links=()):
    """datasets: [{key, bundle, name?, blurb?, view?}, ...]; links: [(label, url), ...] shown on the landing page."""
    app = FastAPI(title=title, docs_url=None, redoc_url=None)
    cards = []
    for d in datasets:
        key = d["key"]; view = d.get("view")
        if view and not os.path.exists(view): view = None
        sub = create_app(d["bundle"], threads=threads, view_file=view, view_writable=False, memory_limit=memory_limit)
        m = sub.state.meta; f = m["fate"]; tot = max(1, f["keep"] + f["move"] + f["drop"])
        cards.append(dict(key=key, name=d.get("name") or m["dataset"], blurb=d.get("blurb", ""), n_cells=m["n_cells"], n_tx=m["n_tx"],
                          n_genes=m["n_genes"], keep=100 * f["keep"] / tot, move=100 * f["move"] / tot, drop=100 * f["drop"] / tot))
        app.add_api_route(f"/d/{key}", _redirect(f"/d/{key}/"), methods=["GET"], include_in_schema=False)
        app.mount(f"/d/{key}", sub)
    intro = intro or ("CellDot corrects the cell assignment of every molecule in an imaging-based spatial transcriptomics section, "
                      "deciding for each one whether it stays, moves to a neighbouring cell or is removed as ambient background. "
                      "Open a dataset to browse the section and the fate of each molecule.")
    card_html = "".join(
        f'<a class="card" href="d/{c["key"]}/"><h2>{html.escape(c["name"])}</h2><div class="blurb">{html.escape(c["blurb"])}</div>'
        f'<div class="nums"><b>{c["n_cells"]:,}</b> cells · <b>{c["n_tx"]:,}</b> molecules · <b>{c["n_genes"]:,}</b> genes</div>'
        f'<div class="fate"><span style="width:{c["keep"]:.1f}%;background:var(--keep)"></span><span style="width:{c["move"]:.1f}%;background:var(--move)"></span>'
        f'<span style="width:{c["drop"]:.1f}%;background:var(--drop)"></span></div>'
        f'<div class="legend"><span><span class="sw" style="background:var(--keep)"></span>keep {c["keep"]:.1f}%</span>'
        f'<span><span class="sw" style="background:var(--move)"></span>move {c["move"]:.1f}%</span><span><span class="sw" style="background:var(--drop)"></span>drop {c["drop"]:.1f}%</span></div>'
        f'<div class="open">open the viewer →</div></a>' for c in cards)
    link_html = "".join(f'<a href="{html.escape(u)}">{html.escape(l)}</a>' for l, u in links)
    page = PAGE.format(title=html.escape(title), intro=html.escape(intro), cards=card_html, links=link_html)

    @app.get("/", include_in_schema=False)
    def index(): return HTMLResponse(page)

    @app.get("/health", include_in_schema=False)
    def health(): return JSONResponse({"ok": True, "datasets": [c["key"] for c in cards]})

    app.state.cards = cards
    return app
