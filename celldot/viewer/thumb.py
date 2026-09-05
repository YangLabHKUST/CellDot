"""Render a picture of a bundle's opening view (cells by type, molecules of the view's genes by fate) without a browser.

    celldot-view-thumb <bundle dir> [--view celldot_view.json] [--out thumb.jpg] [--width 800] [--height 500]

Used for the demo's landing-page cards; the bbox comes from the view file (x, y, width_um) or, without one, from a
1.5 mm window at the centre of the tissue.
"""
import argparse, json, os
import numpy as np
from PIL import Image, ImageDraw

BG = (11, 13, 18); FATE = {0: (138, 143, 152), 1: (47, 143, 224), 2: (229, 72, 77)}


def hexrgb(h): return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def render(bundle, view=None, out=None, width=800, height=500, opacity=0.6, gene_dots=True, log=print):
    import duckdb
    meta = json.load(open(os.path.join(bundle, "meta.json"))); genes = json.load(open(os.path.join(bundle, "genes.json")))["names"]
    e = meta["extent"]; v = {}
    if view and os.path.exists(view):
        d = json.load(open(view)).get("default") or {}
        v = d if isinstance(d, dict) else {}
    cx = float(v.get("x", (e["xmin"] + e["xmax"]) / 2)); cy = float(v.get("y", (e["ymin"] + e["ymax"]) / 2))
    w = float(v.get("width_um", 1500)); h = w * height / width; op = max(float(v.get("opacity", opacity)), 0.7)   # covers stay vivid
    x0, x1, y0, y1 = cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2
    sx = width / w; sy = height / h
    con = duckdb.connect(); con.execute("PRAGMA threads=4")
    cells = os.path.join(bundle, "cells.parquet").replace("'", "''")
    q = con.execute(f"SELECT type, vx, vy FROM read_parquet('{cells}') WHERE nv > 0 AND xmax >= {x0} AND xmin <= {x1} AND ymax >= {y0} AND ymin <= {y1}")
    t = q.to_arrow_table() if hasattr(q, "to_arrow_table") else q.fetch_arrow_table()
    cols = [hexrgb(c) for c in meta["type_colors"]]
    base = Image.new("RGB", (width, height), BG); layer = Image.new("RGBA", (width, height), (0, 0, 0, 0)); dr = ImageDraw.Draw(layer)
    typ = t.column("type").to_numpy(); vx = t.column("vx").to_pylist(); vy = t.column("vy").to_pylist()
    a = int(255 * op)
    for k in range(len(typ)):
        c = cols[typ[k]] if 0 <= typ[k] < len(cols) else (70, 74, 84)
        pts = [((x - x0) * sx, (y - y0) * sy) for x, y in zip(vx[k], vy[k])]
        if len(pts) >= 3: dr.polygon(pts, fill=c + (a,))
    base = Image.alpha_composite(base.convert("RGBA"), layer)
    n_dots = 0
    gl = [genes.index(g) for g in v.get("genes", []) if g in genes]
    if gene_dots and gl:
        mol = os.path.join(bundle, "molecules_sorted.parquet").replace("'", "''")
        same = 1 if v.get("same", True) else 0
        m = con.execute(f"SELECT x, y, act, same FROM read_parquet('{mol}') WHERE x BETWEEN {x0} AND {x1} AND y BETWEEN {y0} AND {y1} AND gene IN ({','.join(map(str, gl))})").fetchnumpy()
        act = m["act"].astype(int); act = np.where((act == 1) & (m["same"] == 1) & (same == 1), 0, act)
        dots = ImageDraw.Draw(base); r = max(1.0, float(v.get("psize", 2.2)) * 0.9 * width / 1400)
        px = (m["x"] - x0) * sx; py = (m["y"] - y0) * sy
        for order in (0, 2, 1):                                    # keep under drop under move
            sel = np.flatnonzero(act == order); c = FATE[order]
            for i in sel: dots.ellipse((px[i] - r, py[i] - r, px[i] + r, py[i] + r), fill=c)
        n_dots = int(len(act))
    out = out or os.path.join(os.path.dirname(os.path.abspath(bundle)), "thumb.jpg")
    base.convert("RGB").save(out, quality=86, optimize=True)
    log(f"thumb: {out} ({width}x{height}) view {w:.0f} um at ({cx:.0f}, {cy:.0f}), {len(typ):,} cells, {n_dots:,} molecules of {len(gl)} gene(s)")
    return out


def main():
    ap = argparse.ArgumentParser(prog="celldot-view-thumb", description="render a bundle's opening view as a JPEG (no browser)")
    ap.add_argument("bundle"); ap.add_argument("--view", help="celldot_view.json (default: next to the bundle)"); ap.add_argument("--out")
    ap.add_argument("--width", type=int, default=800); ap.add_argument("--height", type=int, default=500)
    a = ap.parse_args()
    view = a.view or os.path.join(os.path.dirname(os.path.abspath(a.bundle)), "celldot_view.json")
    render(a.bundle, view=view, out=a.out, width=a.width, height=a.height)


if __name__ == "__main__":
    main()
