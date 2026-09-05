"""CellDot viewer backend: FastAPI + DuckDB over a bundle written by prep.build_bundle.

Binary responses (application/octet-stream): [uint32 header length][JSON header][typed arrays...], header =
{"fields": [[name, dtype, count], ...], ...}; dtypes f32/i32/i16/u8. Only viewport-sized slices ever leave the server.
    GET /api/meta, /api/genes                      bundle metadata, gene names (index = gene code)
    GET /api/cells                                 all cells' attributes (pos order): cx, cy, type, n_in, n_out, n_drop, n_raw, n_clean,
                                                   n_in_same, n_out_same (moves between cells of the same type), in_run
    GET /api/polys?xmin&xmax&ymin&ymax&limit       cell polygons intersecting the box: pos, nv, xy (flat)
    GET /api/transcripts?xmin&xmax&ymin&ymax&genes=1,2&fates=0,1,2&all=0&same=0&limit
                                                   molecules in the box (gene + fate pushdown; reservoir-sampled to limit); each
                                                   molecule carries same=1 when it moved between two cells of one cell type, and
                                                   same=1 in the query makes the fate filter count those moves as 'keep'
    GET /api/expr?gene=&layer=raw|clean            per-cell value of one gene (nonzeros): cell, val
    GET /api/cell?pos=                             one cell: id, attrs, and its gene table (raw, clean)
    GET /api/cell_molecules?pos=&pad=              every molecule that started in or ended in the cell
    GET /api/find?cell_id=                         locate a cell by its original id
    GET /api/summary?xmin&xmax&ymin&ymax           fate counts inside the box (keep, move, drop, move_same)
    GET /api/view                                  the optional view file (celldot_view.json next to cleaned.h5ad): default view,
                                                   bookmarks, type colours; {} when there is none
    POST /api/view                                 write {"default": view} / {"bookmark": {...}} / {"remove_bookmark": name} into
                                                   that file (only when the server is bound to localhost)
"""
import os, json, struct, threading, numpy as np
from fastapi import FastAPI, Response, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from .prep import BUNDLE_VERSION

HERE = os.path.dirname(os.path.abspath(__file__))
DT = {"float32": "f32", "int32": "i32", "int16": "i16", "uint8": "u8", "int8": "i8", "uint32": "u32"}


def pack(fields, **extra):
    hdr = {"fields": [[n, DT[str(a.dtype)], int(a.size)] for n, a in fields]}; hdr.update(extra)
    h = json.dumps(hdr).encode()
    return Response(content=b"".join([struct.pack("<I", len(h)), h] + [np.ascontiguousarray(a).tobytes() for _, a in fields]),
                    media_type="application/octet-stream")


def _ints(s, lo, hi):
    out = []
    for p in (s or "").split(","):
        p = p.strip()
        if p.lstrip("-").isdigit() and lo <= int(p) <= hi: out.append(int(p))
    return out


def create_app(bundle, threads=4, view_file=None, view_writable=False, memory_limit=None):
    """view_file: optional JSON with the opening view / bookmarks / type colours (absent for ordinary runs -> built-in
    defaults); view_writable lets the page save into it (celldot-view enables this only for a localhost server);
    memory_limit caps DuckDB's memory for this bundle (e.g. "3GB" when several bundles share one small server)."""
    import duckdb
    META = json.load(open(os.path.join(bundle, "meta.json"))); GENES = json.load(open(os.path.join(bundle, "genes.json")))
    if META.get("bundle_version", 1) != BUNDLE_VERSION:
        raise RuntimeError(f"viewer bundle {bundle} was built by another CellDot version; rebuild it (celldot-view --rebuild)")
    P = {k: os.path.join(bundle, v).replace("'", "''") for k, v in dict(cells="cells.parquet", mol="molecules_sorted.parquet",
                                                                     eg="expr_by_gene.parquet", ec="expr_by_cell.parquet").items()}
    con = duckdb.connect(); con.execute(f"PRAGMA threads={threads}"); LOCK = threading.Lock()
    if memory_limit: con.execute(f"SET memory_limit='{memory_limit}'")
    def q(sql, params=None):
        with LOCK: return con.execute(sql, params or []).fetchnumpy()
    def q_arrow(sql):
        with LOCK: return con.execute(sql).fetch_arrow_table()
    def one(sql, params=None):
        with LOCK: return con.execute(sql, params or []).fetchone()

    # all-cell attribute arrays, pos order (small: ~30 bytes/cell)
    A = q(f"SELECT pos, cx, cy, type, n_in, n_out, n_drop, n_raw, n_clean, n_in_same, n_out_same, in_run FROM read_parquet('{P['cells']}') ORDER BY pos")
    NT = int(A["pos"].size); assert (A["pos"] == np.arange(NT)).all()
    CX = np.asarray(A["cx"], np.float32); CY = np.asarray(A["cy"], np.float32)
    CELLS_BLOB = pack([("cx", CX), ("cy", CY), ("type", np.asarray(A["type"], np.int16)),
                       ("n_in", np.asarray(A["n_in"], np.int32)), ("n_out", np.asarray(A["n_out"], np.int32)), ("n_drop", np.asarray(A["n_drop"], np.int32)),
                       ("n_raw", np.asarray(A["n_raw"], np.int32)), ("n_clean", np.asarray(A["n_clean"], np.int32)),
                       ("n_in_same", np.asarray(A["n_in_same"], np.int32)), ("n_out_same", np.asarray(A["n_out_same"], np.int32)),
                       ("in_run", np.asarray(A["in_run"], np.uint8))], n=NT, n_run=META["n_cells"])
    app = FastAPI(title=f"CellDot viewer: {META['dataset']}")

    @app.get("/api/meta")
    def meta(): return JSONResponse(META)

    @app.get("/api/genes")
    def genes(): return JSONResponse(GENES)

    @app.get("/api/cells")
    def cells(): return CELLS_BLOB

    @app.get("/api/polys")
    def polys(xmin: float, xmax: float, ymin: float, ymax: float, limit: int = 80000):
        base = (f"FROM read_parquet('{P['cells']}') WHERE nv > 0 AND xmax >= {xmin} AND xmin <= {xmax} AND ymax >= {ymin} AND ymin <= {ymax}")
        n = int(one(f"SELECT count(*) {base}")[0]); sampled = n > limit
        sql = f"SELECT pos, vx, vy {base}" + (f" USING SAMPLE {int(limit)} ROWS (reservoir, 42)" if sampled else "")
        t = q_arrow(sql)
        if t.num_rows == 0:
            return pack([("pos", np.zeros(0, np.int32)), ("nv", np.zeros(0, np.int32)), ("xy", np.zeros(0, np.float32))], n=0, sampled=False)
        vx = t.column("vx").combine_chunks(); vy = t.column("vy").combine_chunks()
        offs = vx.offsets.to_numpy(); fx = vx.values.to_numpy().astype(np.float32); fy = vy.values.to_numpy().astype(np.float32)
        xy = np.empty(2 * fx.size, np.float32); xy[0::2] = fx; xy[1::2] = fy
        return pack([("pos", t.column("pos").to_numpy().astype(np.int32)), ("nv", np.diff(offs).astype(np.int32)), ("xy", xy)], n=int(t.num_rows), sampled=sampled)

    def _tx_where(xmin, xmax, ymin, ymax, genes, fates, all_genes, hide_same=False):
        gl = _ints(genes, 0, GENES["n"] - 1); fl = _ints(fates, 0, 2)
        if not gl and not all_genes: return None
        w = f"x BETWEEN {xmin} AND {xmax} AND y BETWEEN {ymin} AND {ymax}"
        if gl and not all_genes: w += f" AND gene IN ({','.join(map(str, gl))})"
        eff = "(CASE WHEN act = 1 AND same = 1 THEN 0 ELSE act END)" if hide_same else "act"   # same-type moves filtered as 'keep'
        if 0 < len(fl) < 3: w += f" AND {eff} IN ({','.join(map(str, fl))})"
        if not fl: w += " AND act = 9"
        return w

    EMPTY = lambda: pack([("x", np.zeros(0, np.float32)), ("y", np.zeros(0, np.float32)), ("gene", np.zeros(0, np.int32)),
                          ("act", np.zeros(0, np.uint8)), ("old", np.zeros(0, np.int32)), ("new", np.zeros(0, np.int32)),
                          ("same", np.zeros(0, np.uint8))], n=0, sampled=False, total=0)

    def _tx(where, limit):
        base = f"FROM read_parquet('{P['mol']}') WHERE {where}"
        n = int(one(f"SELECT count(*) {base}")[0]); sampled = n > limit
        sql = f"SELECT x, y, gene, act, old, new, same {base}"
        if sampled: sql = f"SELECT * FROM ({sql}) USING SAMPLE {int(limit)} ROWS (reservoir, 42)"
        d = q(sql)
        return pack([("x", np.asarray(d["x"], np.float32)), ("y", np.asarray(d["y"], np.float32)), ("gene", np.asarray(d["gene"], np.int32)),
                     ("act", np.asarray(d["act"], np.uint8)), ("old", np.asarray(d["old"], np.int32)), ("new", np.asarray(d["new"], np.int32)),
                     ("same", np.asarray(d["same"], np.uint8))],
                    n=int(d["x"].size), sampled=sampled, total=n)

    @app.get("/api/transcripts")
    def transcripts(xmin: float, xmax: float, ymin: float, ymax: float, genes: str = "", fates: str = "0,1,2", all: int = 0, same: int = 0,
                    limit: int = 250000):
        w = _tx_where(xmin, xmax, ymin, ymax, genes, fates, bool(all), bool(same))
        return EMPTY() if w is None else _tx(w, limit)

    @app.get("/api/expr")
    def expr(gene: int, layer: str = "clean"):
        if not 0 <= gene < GENES["n"]: raise HTTPException(400, "bad gene")
        col = "raw" if layer == "raw" else "clean"
        d = q(f"SELECT cell, {col} AS val FROM read_parquet('{P['eg']}') WHERE gene = {int(gene)} AND {col} > 0")
        return pack([("cell", np.asarray(d["cell"], np.int32)), ("val", np.asarray(d["val"], np.int32))], gene=gene, layer=col)

    @app.get("/api/cell")
    def cell(pos: int):
        if not 0 <= pos < NT: raise HTTPException(404, "no such cell")
        r = one(f"SELECT cell_id, type, cx, cy, n_in, n_out, n_drop, n_raw, n_clean, in_run, nv, n_in_same, n_out_same "
                f"FROM read_parquet('{P['cells']}') WHERE pos = {int(pos)}")
        g = q(f"SELECT gene, raw, clean FROM read_parquet('{P['ec']}') WHERE cell = {int(pos)} ORDER BY clean DESC, raw DESC") if r[9] else {"gene": [], "raw": [], "clean": []}
        return JSONResponse({"pos": pos, "cell_id": r[0], "type": META["types"][r[1]] if r[1] >= 0 else None, "cx": float(r[2]), "cy": float(r[3]),
                             "n_in": int(r[4]), "n_out": int(r[5]), "n_drop": int(r[6]), "n_raw": int(r[7]), "n_clean": int(r[8]), "in_run": bool(r[9]),
                             "has_polygon": int(r[10]) > 0, "n_in_same": int(r[11]), "n_out_same": int(r[12]),
                             "genes": [[GENES["names"][int(a)], int(b), int(c)] for a, b, c in zip(g["gene"], g["raw"], g["clean"])]})

    @app.get("/api/cell_molecules")
    def cell_molecules(pos: int, pad: float = 80.0):
        if not 0 <= pos < NT: raise HTTPException(404, "no such cell")
        cx, cy = float(CX[pos]), float(CY[pos])
        return _tx(f"x BETWEEN {cx - pad} AND {cx + pad} AND y BETWEEN {cy - pad} AND {cy + pad} AND (old = {int(pos)} OR new = {int(pos)})", 500000)

    @app.get("/api/find")
    def find(cell_id: str):
        r = one(f"SELECT pos, cx, cy, in_run FROM read_parquet('{P['cells']}') WHERE cell_id = ?", [cell_id])
        if r is None: raise HTTPException(404, f"cell_id {cell_id!r} not found")
        return JSONResponse({"pos": int(r[0]), "cx": float(r[1]), "cy": float(r[2]), "in_run": bool(r[3])})

    @app.get("/api/summary")
    def summary(xmin: float, xmax: float, ymin: float, ymax: float):
        d = q(f"SELECT act, same, count(*) AS n FROM read_parquet('{P['mol']}') WHERE x BETWEEN {xmin} AND {xmax} AND y BETWEEN {ymin} AND {ymax} GROUP BY act, same")
        c = {}; move_same = 0
        for a, s, n in zip(d["act"], d["same"], d["n"]):
            c[int(a)] = c.get(int(a), 0) + int(n)
            if int(a) == 1 and int(s) == 1: move_same += int(n)
        return JSONResponse({"keep": c.get(0, 0), "move": c.get(1, 0), "drop": c.get(2, 0), "move_same": move_same})

    def read_view():
        if not view_file or not os.path.exists(view_file): return {}
        try:
            d = json.load(open(view_file, encoding="utf-8")); return d if isinstance(d, dict) else {"error": "view file must hold a JSON object"}
        except Exception as e:
            return {"error": f"{os.path.basename(view_file)}: {e}"}

    @app.get("/api/view")
    def view():
        d = read_view(); d["_writable"] = bool(view_writable and view_file); d["_file"] = view_file; return JSONResponse(d)

    @app.post("/api/view")
    async def save_view(req: Request):
        if not (view_writable and view_file): raise HTTPException(403, "the view file is read-only on this server")
        body = await req.json(); cur = read_view()
        for k in ("_writable", "_file", "error"): cur.pop(k, None)
        if "default" in body: cur["default"] = body["default"]
        if "bookmark" in body: cur.setdefault("bookmarks", []).append(body["bookmark"])
        if "remove_bookmark" in body: cur["bookmarks"] = [b for b in cur.get("bookmarks", []) if b.get("name") != body["remove_bookmark"]]
        json.dump(cur, open(view_file, "w", encoding="utf-8"), indent=1); return JSONResponse({"ok": True, "file": view_file})

    @app.get("/")
    def index(): return HTMLResponse(open(os.path.join(HERE, "web", "index.html"), encoding="utf-8").read())

    app.mount("/web", StaticFiles(directory=os.path.join(HERE, "web")), name="web")
    app.state.meta = META
    return app
