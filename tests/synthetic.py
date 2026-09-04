"""Synthetic Xenium-style fixture for CellDot tests: writes outs/{transcripts,cells}.parquet +
cell_feature_matrix.h5 (10x v3 layout), a reference h5ad and a labels.parquet.

Two id styles: 'xenium' (string ids like 'abcdefgh-1', unassigned = 'UNASSIGNED', as Xenium >= 1.x exports)
and 'int' (integer ids 1..N, unassigned = -1, as the 2022 Xenium 1.0 breast-cancer export). Cell ids are
deliberately NOT sorted so that position <-> id bookkeeping is exercised.
"""
import os, numpy as np, pandas as pd, h5py, scipy.sparse as sp, anndata as ad
import pyarrow as pa, pyarrow.parquet as pq
from scipy.spatial import cKDTree


def _xenium_ids(n, rng):
    letters = np.array(list("abcdefghijklmnop")); seen = set(); out = []
    while len(out) < n:
        s = "".join(rng.choice(letters, 8)) + "-1"
        if s not in seen: seen.add(s); out.append(s)
    return np.array(out, dtype=object)


def make_synthetic(root, seed=0, id_style="xenium", n_cells=2500, n_genes=40, n_types=6, box=(1600.0, 1100.0),
                   mol_per_cell=80, contam=0.15, n_soup=30000, n_ctrl=600, unlabelled_frac=0.02, wrong_label_frac=0.03):
    rng = np.random.default_rng(seed); root = str(root); outs = os.path.join(root, "outs"); os.makedirs(outs, exist_ok=True)
    genes = [f"G{i:03d}" for i in range(n_genes)]; types = [f"T{t}" for t in range(n_types)]
    prof = np.full((n_types, n_genes), 0.2)
    for t in range(n_types): prof[t, (t * 5) % n_genes:(t * 5) % n_genes + 5] = 5.0
    prof /= prof.sum(1, keepdims=True)

    # ---- reference (80 cells/type; 10 extra genes absent from the panel) ----
    n_ref = 80 * n_types; ref_t = np.repeat(np.arange(n_types), 80)
    ref_X = np.stack([rng.multinomial(1500, prof[t]) for t in ref_t]); extra = rng.poisson(0.5, size=(n_ref, 10))
    ref = ad.AnnData(X=sp.csr_matrix(np.hstack([ref_X, extra]).astype(np.float32)),
                     obs=pd.DataFrame({"celltype": pd.Categorical([types[t] for t in ref_t])}, index=[f"ref{i}" for i in range(n_ref)]),
                     var=pd.DataFrame(index=genes + [f"X{i}" for i in range(10)]))
    ref.write_h5ad(os.path.join(root, "reference.h5ad"))

    # ---- cells ----
    if id_style == "xenium":
        cid = _xenium_ids(n_cells, rng); unassigned = "UNASSIGNED"
    else:
        cid = rng.permutation(np.arange(1, n_cells + 1)).astype(np.int32); unassigned = -1
    cx = rng.uniform(0, box[0], n_cells); cy = rng.uniform(0, box[1], n_cells)
    area = rng.uniform(60, 200, n_cells); rad = np.sqrt(area / np.pi); ct = rng.integers(0, n_types, n_cells)

    # ---- cellular molecules: host cell + (contam) a neighbour's profile ----
    n_mol = rng.poisson(mol_per_cell, n_cells); host = np.repeat(np.arange(n_cells), n_mol); M = len(host)
    _, nb = cKDTree(np.c_[cx, cy]).query(np.c_[cx, cy], k=2); nbr = nb[:, 1]
    src = np.where(rng.random(M) < contam, nbr[host], host)
    gene = np.zeros(M, np.int64)
    for t in range(n_types):
        m = ct[src] == t; gene[m] = rng.choice(n_genes, size=int(m.sum()), p=prof[t])
    mx = np.clip(cx[host] + rng.normal(0, 0.6 * rad[host]), 0, box[0]); my = np.clip(cy[host] + rng.normal(0, 0.6 * rad[host]), 0, box[1])
    qv = rng.uniform(15, 40, M).astype(np.float32)
    # ---- soup (unassigned) + control features ----
    soup_p = rng.dirichlet(np.full(n_genes, 0.3)); sg = rng.choice(n_genes, size=n_soup, p=soup_p)
    sx = rng.uniform(0, box[0], n_soup); sy = rng.uniform(0, box[1], n_soup); sqv = rng.uniform(15, 40, n_soup).astype(np.float32)
    ctrl_names = np.array(["NegControlCodeword_0500", "NegControlCodeword_0501", "NegControlProbe_00001", "BLANK_0001"], dtype=object)
    cn = ctrl_names[rng.integers(0, 4, n_ctrl)]; c_host = rng.integers(-1, n_cells, n_ctrl)   # -1 -> unassigned
    cxx = np.where(c_host >= 0, cx[np.clip(c_host, 0, None)], rng.uniform(0, box[0], n_ctrl)); cyy = np.where(c_host >= 0, cy[np.clip(c_host, 0, None)], rng.uniform(0, box[1], n_ctrl))

    feat = np.concatenate([np.array(genes, dtype=object)[gene], np.array(genes, dtype=object)[sg], cn])
    x = np.concatenate([mx, sx, cxx]).astype(np.float32); y = np.concatenate([my, sy, cyy]).astype(np.float32)
    q = np.concatenate([qv, sqv, rng.uniform(15, 40, n_ctrl).astype(np.float32)])
    if id_style == "xenium":
        cell = np.concatenate([cid[host], np.full(n_soup, unassigned, dtype=object), np.where(c_host >= 0, cid[np.clip(c_host, 0, None)], unassigned)]).astype(object)
    else:
        cell = np.concatenate([cid[host], np.full(n_soup, unassigned, np.int32), np.where(c_host >= 0, cid[np.clip(c_host, 0, None)], unassigned)]).astype(np.int32)
    perm = rng.permutation(len(feat))
    tx = pd.DataFrame({"transcript_id": np.arange(len(feat), dtype=np.int64), "cell_id": cell[perm], "overlaps_nucleus": rng.integers(0, 2, len(feat)).astype(np.int8),
                       "feature_name": feat[perm], "x_location": x[perm], "y_location": y[perm], "z_location": rng.uniform(0, 10, len(feat)).astype(np.float32), "qv": q[perm]})
    pq.write_table(pa.Table.from_pandas(tx, preserve_index=False), os.path.join(outs, "transcripts.parquet"), row_group_size=50000)

    # ---- cells.parquet (Xenium 1.0-style columns; NO segmentation_method / codeword columns) ----
    n_tx_cell = np.bincount(host, minlength=n_cells); n_ctrl_cell = np.bincount(c_host[c_host >= 0], minlength=n_cells)
    cells = pd.DataFrame({"cell_id": cid, "x_centroid": cx, "y_centroid": cy, "transcript_counts": n_tx_cell.astype(np.int32),
                          "control_probe_counts": (n_ctrl_cell // 2).astype(np.int32), "control_codeword_counts": (n_ctrl_cell - n_ctrl_cell // 2).astype(np.int32),
                          "total_counts": (n_tx_cell + n_ctrl_cell).astype(np.int32), "cell_area": area.astype(np.float32), "nucleus_area": (0.3 * area).astype(np.float32)})
    cells.to_parquet(os.path.join(outs, "cells.parquet"), index=False)

    # ---- cell_boundaries.parquet (12-vertex jittered rings; Xenium column names) ----
    nvert = 12; ang = np.linspace(0, 2 * np.pi, nvert, endpoint=False)
    rr = rad[:, None] * (1.0 + 0.15 * rng.standard_normal((n_cells, nvert)))
    bx = (cx[:, None] + rr * np.cos(ang)).astype(np.float32); by = (cy[:, None] + rr * np.sin(ang)).astype(np.float32)
    pd.DataFrame({"cell_id": np.repeat(cid, nvert), "vertex_x": bx.ravel(), "vertex_y": by.ravel(),
                  "label_id": np.repeat(np.arange(1, n_cells + 1, dtype=np.int32), nvert)}).to_parquet(os.path.join(outs, "cell_boundaries.parquet"), index=False)

    # ---- cell_feature_matrix.h5 (10x v3 layout: features x barcodes CSC) + 3 control features ----
    counts = sp.coo_matrix((np.ones(M, np.int32), (host, gene)), shape=(n_cells, n_genes)).tocsr()
    ctrl_feats = [("NegControlCodeword_0500", "Negative Control Codeword"), ("NegControlProbe_00001", "Negative Control Probe"), ("UnassignedCodeword_0001", "Unassigned Codeword")]
    full = sp.hstack([counts, sp.csr_matrix((n_cells, len(ctrl_feats)), dtype=np.int32)]).tocsc()   # cells x features
    fc = sp.csc_matrix(full.T)   # features x cells... 10x stores the matrix as (features, barcodes) CSC = cells are columns
    fc = sp.csc_matrix(full.T.tocsr())
    with h5py.File(os.path.join(outs, "cell_feature_matrix.h5"), "w") as f:
        g = f.create_group("matrix")
        g.create_dataset("barcodes", data=np.array([str(c) for c in cid], dtype="S"))
        g.create_dataset("data", data=fc.data.astype(np.int32)); g.create_dataset("indices", data=fc.indices.astype(np.int64))
        g.create_dataset("indptr", data=fc.indptr.astype(np.int64)); g.create_dataset("shape", data=np.array(fc.shape, dtype=np.int32))
        fg = g.create_group("features"); names = genes + [n for n, _ in ctrl_feats]; ftype = ["Gene Expression"] * n_genes + [t for _, t in ctrl_feats]
        fg.create_dataset("id", data=np.array([f"ENSG{i:08d}" for i in range(len(names))], dtype="S")); fg.create_dataset("name", data=np.array(names, dtype="S"))
        fg.create_dataset("feature_type", data=np.array(ftype, dtype="S")); fg.create_dataset("genome", data=np.array(["Synthetic"] * len(names), dtype="S"))
        fg.create_dataset("_all_tag_keys", data=np.array(["genome"], dtype="S"))

    # ---- labels (a few unlabelled, a few wrong) ----
    keep = rng.random(n_cells) >= unlabelled_frac
    lab_t = np.where(rng.random(n_cells) < wrong_label_frac, rng.integers(0, n_types, n_cells), ct)
    lab = pd.DataFrame({"cell_id": cid[keep], "celltype": [types[t] for t in lab_t[keep]], "prob": rng.uniform(0.5, 1.0, int(keep.sum())).astype(np.float32)})
    lab.to_parquet(os.path.join(root, "labels.parquet"), index=False)
    return dict(outs=outs, reference=os.path.join(root, "reference.h5ad"), labels=os.path.join(root, "labels.parquet"), boundaries=os.path.join(outs, "cell_boundaries.parquet"),
                n_cells=n_cells, n_mol=M, n_soup=n_soup, unassigned=unassigned)
