"""One server, several CellDot results: the public demo.

    from celldot.viewer.multi import create_multi_app
    app = create_multi_app([
        dict(key="BreastCancer", name="Breast cancer", blurb="10x Xenium, FFPE section",
             bundle="/data/BreastCancer/viewer_bundle", view="/data/BreastCancer/celldot_view.json"),
        ...])                                   # then uvicorn.run(app, host="0.0.0.0", port=7860)

A landing page at / lists the datasets (with <bundle parent>/thumb.jpg as the card image when present); each viewer
lives at /d/<key>/ (the viewer page uses relative URLs, so it runs unchanged under the prefix). View files are read-only
here (no "set as default view" from the page).
"""
import html, os
from fastapi import FastAPI
from fastapi import Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, FileResponse
from .server import create_app

PAGE = """<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<link rel="icon" href="data:image/svg+xml,{favicon}">
<style>
  :root {{ --bg:#0b0d12; --panel:#141821; --line:#262c38; --fg:#e8eaf0; --mut:#98a2b3; --acc:#8FC0CD; --keep:#8a8f98; --move:#2f8fe0; --drop:#e5484d; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 -apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }}
  main {{ max-width:1360px; margin:0 auto; padding:56px 24px 48px; }}
  header {{ display:flex; align-items:center; gap:18px; margin-bottom:8px; }}
  header svg {{ width:56px; height:56px; flex:none; }}
  h1 {{ font-size:30px; margin:0; letter-spacing:-.01em; font-weight:650; }}
  h1 span {{ color:var(--acc); }}
  .lead {{ color:var(--mut); margin:0 0 30px 74px; font-size:15px; }}
  .cards {{ display:grid; grid-template-columns:repeat(4, 1fr); gap:18px; }}
  @media (max-width:1180px) {{ .cards {{ grid-template-columns:repeat(2, 1fr); }} }}
  @media (max-width:640px) {{ .cards {{ grid-template-columns:1fr; }} }}
  a.card {{ display:block; background:var(--panel); border:1px solid var(--line); border-radius:14px; overflow:hidden; color:inherit; text-decoration:none; transition:border-color .15s, transform .15s; }}
  a.card:hover {{ border-color:var(--acc); transform:translateY(-2px); }}
  a.card:focus-visible {{ outline:2px solid var(--acc); outline-offset:2px; }}
  .thumb {{ aspect-ratio:16/10; background:#0e1117 center/cover no-repeat; border-bottom:1px solid var(--line); }}
  .body {{ padding:14px 16px 16px; }}
  .card h2 {{ font-size:17px; margin:0 0 2px; font-weight:600; }}
  .blurb {{ color:var(--mut); font-size:12.5px; }}
  .nums {{ font-variant-numeric:tabular-nums; font-size:12.5px; margin:10px 0 8px; color:var(--mut); }}
  .nums b {{ color:var(--fg); font-weight:600; }}
  .fate {{ display:flex; height:5px; border-radius:3px; overflow:hidden; background:#222838; }}
  .fate span {{ display:block; height:100%; }}
  .legend {{ display:flex; gap:12px; font-size:11.5px; color:var(--mut); margin-top:6px; }} .legend span {{ white-space:nowrap; }}
  .sw {{ width:8px; height:8px; border-radius:2px; display:inline-block; margin-right:4px; vertical-align:middle; }}
  footer {{ margin-top:34px; font-size:12.5px; color:var(--mut); }} footer a {{ color:var(--acc); text-decoration:none; margin-right:18px; }}
  @media (prefers-reduced-motion: reduce) {{ a.card {{ transition:none; }} }}
</style>
<main>
  <header>{mark}<h1><span>Cell</span>Dot viewer</h1></header>
  <p class="lead">{intro}</p>
  <div class="cards">{cards}</div>
  <footer>{links}</footer>
</main>
"""

MARK = ('<svg viewBox="0 0 64 64" role="img" aria-label="CellDot"><path d="M32 11 C44 11 53 19 53 31 C53 43 45 55 32 55 C20 55 11 46 11 34 C11 22 20 11 32 11 Z" '
        'fill="none" stroke="#8FC0CD" stroke-width="3.2" stroke-linejoin="round"/><circle cx="24" cy="28" r="2.6" fill="#8FC0CD"/><circle cx="33.5" cy="21.5" r="2.6" fill="#8FC0CD"/>'
        '<circle cx="26.5" cy="42" r="2.6" fill="#8FC0CD"/><circle cx="38.5" cy="45" r="2.6" fill="#8FC0CD"/><path d="M55.5 13.5 C51 17 47 22 43 28" fill="none" stroke="#D9A441" '
        'stroke-width="2.6" stroke-linecap="round"/><path d="M43 28 l4.5 -2.2 M43 28 l0.3 -5.0" fill="none" stroke="#D9A441" stroke-width="2.6" stroke-linecap="round"/>'
        '<circle cx="58.5" cy="10" r="3.2" fill="#D9A441"/></svg>')
FAVICON = ("%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Cpath d='M32 8 C46 8 56 18 56 32 C56 46 46 56 32 56 C18 56 8 47 8 33 C8 19 18 8 32 8 Z' "
           "fill='%2317697F'/%3E%3Ccircle cx='39' cy='25.5' r='6.4' fill='%23D9A441'/%3E%3C/svg%3E")


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
    intro = intro or "The fate of every molecule (keep · move · drop) in a CellDot-corrected section. Pick a dataset."
    card_html = lambda: "".join(
        f'<a class="card" href="d/{c["key"]}/"><div class="thumb" style="background-image:url({thumb_url(c["key"])})"></div><div class="body">'
        f'<h2>{html.escape(c["name"])}</h2><div class="blurb">{html.escape(c["blurb"])}</div>'
        f'<div class="nums"><b>{c["n_cells"]:,}</b> cells · <b>{c["n_tx"]:,}</b> molecules · <b>{c["n_genes"]:,}</b> genes</div>'
        f'<div class="fate"><span style="width:{c["keep"]:.1f}%;background:var(--keep)"></span><span style="width:{c["move"]:.1f}%;background:var(--move)"></span>'
        f'<span style="width:{c["drop"]:.1f}%;background:var(--drop)"></span></div>'
        f'<div class="legend"><span><span class="sw" style="background:var(--keep)"></span>keep {c["keep"]:.1f}%</span>'
        f'<span><span class="sw" style="background:var(--move)"></span>move {c["move"]:.1f}%</span><span><span class="sw" style="background:var(--drop)"></span>drop {c["drop"]:.1f}%</span></div>'
        f'</div></a>' for c in cards)
    link_html = "".join(f'<a href="{html.escape(u)}">{html.escape(l)}</a>' for l, u in links)
    def page():
        return PAGE.format(title=html.escape(title), intro=html.escape(intro), cards=card_html(), links=link_html, mark=MARK, favicon=FAVICON)

    @app.get("/", include_in_schema=False)
    def index(): return HTMLResponse(page())

    THUMBS = {d["key"]: os.path.join(os.path.dirname(os.path.abspath(d["bundle"])), "thumb.jpg") for d in datasets}
    def thumb_url(key):
        p = THUMBS.get(key)
        return f"thumb/{key}.jpg?v={int(os.path.getmtime(p))}" if p and os.path.exists(p) else ""

    @app.get("/thumb/{key}.jpg", include_in_schema=False)
    def thumb(key: str):
        p = THUMBS.get(key)
        if not p or not os.path.exists(p): return Response(status_code=404)
        return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})

    @app.get("/health", include_in_schema=False)
    def health(): return JSONResponse({"ok": True, "datasets": [c["key"] for c in cards]})

    app.state.cards = cards
    return app
