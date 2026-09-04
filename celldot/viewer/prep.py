"""Build the viewer bundle from a CellDot run + the platform's cell boundaries.

Reads (no anndata needed: h5py + pyarrow + duckdb)
    <run>/cleaned.h5ad        layers raw / celldot, obs (cell_id index, type, centroid, per-cell fate counts), var
    <run>/molecules.parquet   per-molecule fate: x, y, gene, old_host, new_host, action   (cell_id strings)
    cell_boundaries.parquet   cell_id, vertex_x, vertex_y  (any platform export with these columns)
Writes <bundle>/
    meta.json                 counts, extent, types + colours, fate totals, provenance
    genes.json                gene names (index == the integer gene code used below)
    cells.parquet             one row per cell (run cells first, in obs order = pos; then boundary-only cells):
                              pos, cell_id, type, cx, cy, n_in, n_out, n_drop, n_raw, n_clean, in_run, bbox, polygon
                              (vx/vy lists) — TILE-SORTED so a viewport query touches few row groups
    molecules_sorted.parquet  x, y, gene(code), act(0 keep/1 move/2 drop), old(pos), new(pos; -1 dropped) — TILE-SORTED
    expr_by_gene.parquet      gene, cell, raw, clean   sorted by gene  (per-gene cell fills)
    expr_by_cell.parquet      cell, gene, raw, clean   sorted by cell  (per-cell gene tables)
"""
import os, json, time, tempfile, numpy as np, h5py, scipy.sparse as sp
import pyarrow as pa, pyarrow.parquet as pq

PALETTE = ["#e6194b", "#3cb44b", "#d4a017", "#4363d8", "#f58231", "#911eb4", "#2bb5c4", "#f032e6", "#7cae00",
           "#d68aa8", "#469990", "#8d6fc7", "#9a6324", "#b0a030", "#800000", "#3a9d6e", "#000075", "#e07b39",
           "#5b8fd6", "#c94c9d", "#7f7f7f", "#a1c9f4", "#ff9f9b", "#b5e550", "#ffcc00", "#6a3d9a", "#33a02c", "#1f78b4"]


def _s(x):
    return x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x)


def read_run(h5_path):
    """The parts of cleaned.h5ad the viewer needs, read with h5py."""
    with h5py.File(h5_path, "r") as f:
        def load_sparse(g):
            enc = _s(g.attrs.get("encoding-type", "csr_matrix")); shape = tuple(int(v) for v in g.attrs["shape"])
            args = (g["data"][:], g["indices"][:], g["indptr"][:])
            return sp.csc_matrix(args, shape=shape).tocsr() if "csc" in enc else sp.csr_matrix(args, shape=shape)
        obs = f["obs"]; ikey = _s(obs.attrs["_index"])
        cell_id = np.array([_s(c) for c in obs[ikey][:]], dtype=object); N = len(cell_id)
        def col(name, default=None):
            if name not in obs: return default
            o = obs[name]
            if isinstance(o, h5py.Group):
                cats = np.array([_s(c) for c in o["categories"][:]], dtype=object); return cats[o["codes"][:]]
            a = o[:]
            return np.array([_s(v) for v in a], dtype=object) if a.dtype.kind in "SOU" else a
        typ = col("type", np.array(["cell"] * N, dtype=object)).astype(str)
        cx = col("x_centroid").astype(np.float64); cy = col("y_centroid").astype(np.float64)
        n_in = col("n_moved_in", np.zeros(N)); n_out = col("n_moved_out", np.zeros(N)); n_drop = col("n_dropped", np.zeros(N))
        var = f["var"]; vkey = _s(var.attrs["_index"]); genes = [_s(g) for g in var[vkey][:]]
        raw = load_sparse(f["layers/raw"]) if "layers/raw" in f else load_sparse(f["X"])
        lay = "layers/celldot" if "layers/celldot" in f else ("layers/spdenoise" if "layers/spdenoise" in f else None)
        assert lay, "cleaned.h5ad has no 'celldot' layer"
        clean = load_sparse(f[lay])
        prov = {}
        for key in ("celldot", "spd"):
            if f"uns/{key}" in f:
                g = f[f"uns/{key}"]
                for k in ("dataset", "run_id", "version", "n_tx"):
                    if k in g:
                        v = g[k][()]; v = v.decode() if isinstance(v, bytes) else (v.item() if isinstance(v, np.generic) else v)
                        prov[k] = v
                break
    return dict(cell_id=cell_id, typ=typ, cx=cx, cy=cy, n_in=np.asarray(n_in, np.int64), n_out=np.asarray(n_out, np.int64),
                n_drop=np.asarray(n_drop, np.int64), genes=genes, raw=raw, clean=clean, prov=prov)


def read_boundaries(path):
    """cell_id (str), and per-cell contiguous vertex runs (start/end offsets into vx/vy)."""
    t = pq.read_table(path, columns=["cell_id", "vertex_x", "vertex_y"])
    cid = np.asarray(t.column("cell_id").cast(pa.string()).to_numpy(zero_copy_only=False), dtype=object)
    vx = t.column("vertex_x").to_numpy().astype(np.float32); vy = t.column("vertex_y").to_numpy().astype(np.float32)
    if len(cid) == 0: return np.array([], dtype=object), np.zeros(0, np.int64), np.zeros(0, np.int64), vx, vy
    brk = np.flatnonzero(np.r_[True, cid[1:] != cid[:-1]])
    if len(set(cid[brk].tolist())) != len(brk):                      # ids not contiguous -> stable sort by id
        order = np.argsort(cid, kind="stable"); cid = cid[order]; vx = vx[order]; vy = vy[order]
        brk = np.flatnonzero(np.r_[True, cid[1:] != cid[:-1]])
    end = np.r_[brk[1:], len(cid)]
    return cid[brk], brk, end, vx, vy


def build_bundle(run_dir, boundaries, out_dir, tile=100.0, threads=4, memory_limit="8GB", log=None):
    import duckdb
    log = log or (lambda *a: print(*a, flush=True))
    t0 = time.time(); L = lambda *a: log(f"[viewer-prep {time.time()-t0:6.1f}s]", *a)
    os.makedirs(out_dir, exist_ok=True)
    h5 = os.path.join(run_dir, "cleaned.h5ad"); mol = os.path.join(run_dir, "molecules.parquet")
    R = read_run(h5); N = len(R["cell_id"]); G = len(R["genes"])
    L(f"run: {N:,} cells x {G} genes | provenance {R['prov']}")
    json.dump({"names": R["genes"], "n": G}, open(os.path.join(out_dir, "genes.json"), "w"))

    # ---------------- cells (run cells in obs order, then boundary-only cells) ----------------
    bid, bs, be, vx, vy = read_boundaries(boundaries)
    poly_of = {c: k for k, c in enumerate(bid.tolist())}
    L(f"boundaries: {len(bid):,} cells, {len(vx):,} vertices")
    types, counts = np.unique(R["typ"], return_counts=True)
    types = [t for t, _ in sorted(zip(types.tolist(), counts.tolist()), key=lambda z: -z[1])]
    tcode = {t: i for i, t in enumerate(types)}
    run_ids = set(R["cell_id"].tolist()); extra_ids = [c for c in bid.tolist() if c not in run_ids] if len(bid) else []
    pos_ids = np.concatenate([R["cell_id"], np.array(extra_ids, dtype=object)]) if extra_ids else R["cell_id"]
    NT = len(pos_ids); E = NT - N
    ptype = np.concatenate([np.array([tcode[t] for t in R["typ"]], np.int16), np.full(E, -1, np.int16)])
    nv = np.zeros(NT, np.int64); st = np.zeros(NT, np.int64)
    for p in range(NT):
        k = poly_of.get(pos_ids[p])
        if k is not None: st[p] = bs[k]; nv[p] = be[k] - bs[k]
    has = nv > 0
    cxa = np.zeros(NT); cya = np.zeros(NT); cxa[:N] = R["cx"]; cya[:N] = R["cy"]
    xmin = np.zeros(NT); xmax = np.zeros(NT); ymin = np.zeros(NT); ymax = np.zeros(NT)
    for p in np.flatnonzero(has):
        sx = vx[st[p]:st[p] + nv[p]]; sy = vy[st[p]:st[p] + nv[p]]
        xmin[p] = sx.min(); xmax[p] = sx.max(); ymin[p] = sy.min(); ymax[p] = sy.max()
        if p >= N: cxa[p] = sx.mean(); cya[p] = sy.mean()
    r0 = 5.0
    for arr, c, s in ((xmin, cxa, -r0), (xmax, cxa, r0), (ymin, cya, -r0), (ymax, cya, r0)): arr[~has] = c[~has] + s
    n_raw = np.concatenate([np.asarray(R["raw"].sum(1)).ravel(), np.zeros(E)]).astype(np.int32)
    n_clean = np.concatenate([np.asarray(R["clean"].sum(1)).ravel(), np.zeros(E)]).astype(np.int32)
    n_in = np.concatenate([R["n_in"], np.zeros(E)]).astype(np.int32); n_out = np.concatenate([R["n_out"], np.zeros(E)]).astype(np.int32)
    n_drop = np.concatenate([R["n_drop"], np.zeros(E)]).astype(np.int32)
    key = np.floor(cya / tile).astype(np.int64) * (1 << 20) + np.floor(cxa / tile).astype(np.int64)
    order = np.argsort(key, kind="stable")
    cnt = nv[order]; offs = np.r_[0, np.cumsum(cnt)].astype(np.int32)
    idx = np.concatenate([np.arange(st[p], st[p] + nv[p]) for p in order[has[order]]]) if has.any() else np.zeros(0, np.int64)
    cells = pa.table({
        "pos": pa.array(order.astype(np.int32)), "cell_id": pa.array(pos_ids[order].tolist(), pa.string()),
        "type": pa.array(ptype[order]), "cx": pa.array(cxa[order].astype(np.float32)), "cy": pa.array(cya[order].astype(np.float32)),
        "n_in": pa.array(n_in[order]), "n_out": pa.array(n_out[order]), "n_drop": pa.array(n_drop[order]),
        "n_raw": pa.array(n_raw[order]), "n_clean": pa.array(n_clean[order]), "in_run": pa.array((order < N)),
        "nv": pa.array(cnt.astype(np.int32)),
        "xmin": pa.array(xmin[order].astype(np.float32)), "xmax": pa.array(xmax[order].astype(np.float32)),
        "ymin": pa.array(ymin[order].astype(np.float32)), "ymax": pa.array(ymax[order].astype(np.float32)),
        "vx": pa.ListArray.from_arrays(pa.array(offs, pa.int32()), pa.array(vx[idx])),
        "vy": pa.ListArray.from_arrays(pa.array(offs, pa.int32()), pa.array(vy[idx])),
    })
    pq.write_table(cells, os.path.join(out_dir, "cells.parquet"), row_group_size=20000, compression="zstd")
    L(f"cells.parquet: {NT:,} cells ({N:,} in run, {E:,} boundary-only, {int(has[:N].sum()):,} run cells with polygons)")

    # ---------------- expression (raw & clean, union of nonzeros) --------------------------------
    con = duckdb.connect(); con.execute(f"PRAGMA threads={threads}"); con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET temp_directory='{tempfile.mkdtemp(prefix='celldot_view_')}'")
    def coo(M):
        c = M.tocoo(); return pa.table({"gene": c.col.astype(np.int32), "cell": c.row.astype(np.int32), "val": c.data.astype(np.int32)})
    con.register("raw_t", coo(R["raw"])); con.register("clean_t", coo(R["clean"]))
    con.execute("CREATE TABLE expr AS SELECT COALESCE(r.gene, c.gene) AS gene, COALESCE(r.cell, c.cell) AS cell, "
                "COALESCE(r.val, 0) AS raw, COALESCE(c.val, 0) AS clean FROM raw_t r FULL OUTER JOIN clean_t c ON r.gene = c.gene AND r.cell = c.cell")
    con.execute(f"COPY (SELECT * FROM expr ORDER BY gene, cell) TO '{os.path.join(out_dir, 'expr_by_gene.parquet')}' (FORMAT PARQUET, ROW_GROUP_SIZE 262144, COMPRESSION zstd)")
    con.execute(f"COPY (SELECT * FROM expr ORDER BY cell, gene) TO '{os.path.join(out_dir, 'expr_by_cell.parquet')}' (FORMAT PARQUET, ROW_GROUP_SIZE 262144, COMPRESSION zstd)")
    nnz = con.execute("SELECT count(*) FROM expr").fetchone()[0]; con.execute("DROP TABLE expr"); con.unregister("raw_t"); con.unregister("clean_t")
    L(f"expression tables: {nnz:,} (cell, gene) entries")

    # ---------------- molecules: cell_id -> pos, gene name -> code, tile-sorted --------------------
    con.register("cells_map", pa.table({"cell_id": pa.array(R["cell_id"].tolist(), pa.string()), "pos": np.arange(N, dtype=np.int32)}))
    con.register("gene_map", pa.table({"name": pa.array(R["genes"], pa.string()), "code": np.arange(G, dtype=np.int32)}))
    msort = os.path.join(out_dir, "molecules_sorted.parquet")
    con.execute(f"""
COPY (
  SELECT CAST(m.x AS REAL) AS x, CAST(m.y AS REAL) AS y, g.code AS gene,
         CAST(CASE m.action WHEN 'keep' THEN 0 WHEN 'move' THEN 1 ELSE 2 END AS TINYINT) AS act,
         o.pos AS old, COALESCE(n.pos, -1) AS new
  FROM read_parquet('{mol}') m
  JOIN gene_map g ON m.gene = g.name
  JOIN cells_map o ON m.old_host = o.cell_id
  LEFT JOIN cells_map n ON m.new_host = n.cell_id
  ORDER BY CAST(floor(m.y / {tile}) AS INTEGER), CAST(floor(m.x / {tile}) AS INTEGER)
) TO '{msort}' (FORMAT PARQUET, ROW_GROUP_SIZE 200000, COMPRESSION zstd)""")
    n_in_mol = con.execute(f"SELECT count(*) FROM read_parquet('{mol}')").fetchone()[0]
    ntx, x0, x1, y0, y1 = con.execute(f"SELECT count(*), min(x), max(x), min(y), max(y) FROM read_parquet('{msort}')").fetchone()
    assert ntx == n_in_mol, f"{n_in_mol - ntx} molecules reference a gene or cell absent from cleaned.h5ad"
    fate = dict(con.execute(f"SELECT act, count(*) FROM read_parquet('{msort}') GROUP BY act").fetchall()); con.close()
    L(f"molecules_sorted.parquet: {ntx:,} molecules | fate keep {fate.get(0,0):,} move {fate.get(1,0):,} drop {fate.get(2,0):,}")

    ex = dict(xmin=float(min(x0, np.nanmin(xmin))), xmax=float(max(x1, np.nanmax(xmax))), ymin=float(min(y0, np.nanmin(ymin))), ymax=float(max(y1, np.nanmax(ymax))))
    meta = {"dataset": R["prov"].get("dataset", os.path.basename(os.path.abspath(run_dir))), "provenance": R["prov"],
            "n_cells": int(N), "n_cells_total": int(NT), "n_cells_polygon": int(has.sum()), "n_tx": int(ntx), "n_genes": G,
            "extent": ex, "types": types, "type_colors": [PALETTE[i % len(PALETTE)] for i in range(len(types))],
            "type_counts": [int((R["typ"] == t).sum()) for t in types],
            "fate": {"keep": int(fate.get(0, 0)), "move": int(fate.get(1, 0)), "drop": int(fate.get(2, 0))},
            "tile": float(tile), "built": time.strftime("%Y-%m-%d %H:%M:%S"), "run_dir": os.path.abspath(run_dir), "boundaries": os.path.abspath(boundaries)}
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    L(f"DONE -> {out_dir}")
    return meta
