"""Compare a CellDot run against a spdenoise (v0.1.1) run on the SAME inputs — the two must agree exactly.

spdenoise names cells by integer row (cells_index.parquet 'row' -> cell_id; molecules old/new_host = row,
-1 = dropped; molecules gene = var index). CellDot names them by the original cell_id (obs_names) and writes the
input transcripts.parquet back with celldot_cell_id / celldot_fate. This maps the spdenoise output onto the
cell_id scheme and checks every layer, obs column, the multiset of molecule fates and every provenance count.

    python compare_outputs.py --celldot <out_dir> --spdenoise <out_dir>      # default file names
    python compare_outputs.py --celldot-h5ad … --celldot-mol … --spd-h5ad … --spd-mol … --spd-cells-index …
Exit code 0 iff everything matches.
"""
import os, sys, argparse, json, numpy as np, pandas as pd, anndata as ad, pyarrow.parquet as pq


def _same_sparse(a, b):
    a = a.tocsr(); b = b.tocsr()
    return a.shape == b.shape and (a != b).nnz == 0


def compare(cd_h5ad, cd_mol, spd_h5ad, spd_mol, spd_cells_index, cd_prep=None, spd_prep=None, log=print):
    res = {}
    A = ad.read_h5ad(cd_h5ad); B = ad.read_h5ad(spd_h5ad)
    ci = pd.read_parquet(spd_cells_index).sort_values("row").reset_index(drop=True)
    cid_spd = ci["cell_id"].astype(str).values
    res["spd_obs_matches_cells_index"] = bool((B.obs["cell_id"].astype(str).values == cid_spd).all())
    res["obs_names_equal_and_same_order"] = bool(A.n_obs == B.n_obs and (A.obs_names.values.astype(str) == cid_spd).all())
    res["obs_index_name_is_cell_id"] = A.obs.index.name == "cell_id" and "cell_id" not in A.obs.columns
    res["var_names_equal"] = bool(list(A.var_names) == list(B.var_names))
    if not res["obs_names_equal_and_same_order"]:
        log("  obs order differs -> aligning by cell_id"); B = B[pd.Index(cid_spd).get_indexer(A.obs_names.values.astype(str))].copy()
    res["layer_raw==raw"] = _same_sparse(A.layers["raw"], B.layers["raw"])
    res["X==spdenoise"] = _same_sparse(A.X, B.layers["spdenoise"])          # CellDot's X = the corrected counts
    for c in ["x_centroid", "y_centroid", "mu_bg", "n_dropped", "n_moved_out", "n_moved_in", "drop_frac"]:
        res[f"obs.{c}"] = bool(np.array_equal(A.obs[c].values, B.obs[c].values))
    res["obs.type"] = bool((A.obs["type"].astype(str).values == B.obs["type"].astype(str).values).all())
    res["var.lambda0"] = bool(np.array_equal(A.var["lambda0"].values, B.var["lambda0"].values))

    # molecules: CellDot's transcripts.parquet (input rows + celldot_cell_id/celldot_fate) vs spdenoise's molecules.parquet
    # (tile order, integer rows). Both are reduced to (x, y, gene, source, destination, fate) keys and compared as MULTISETS.
    import pyarrow.compute as pc
    ta = pq.read_table(cd_mol, columns=["x_location", "y_location", "feature_name", "cell_id", "celldot_cell_id", "celldot_fate"],
                       filters=pc.field("celldot_fate").isin(["keep", "move", "drop"]))
    mb = pq.read_table(spd_mol, read_dictionary=["action"]).to_pandas()
    res["mol_nrows"] = ta.num_rows == len(mb)
    if res["mol_nrows"]:
        gI = pd.Index(list(B.var_names)); cI = pd.Index(cid_spd)
        def strs(col): return np.asarray(col.cast("string").to_numpy(zero_copy_only=False), dtype=object)
        ka = np.rec.fromarrays([ta.column("x_location").to_numpy().astype(np.float32), ta.column("y_location").to_numpy().astype(np.float32),
                                gI.get_indexer(strs(ta.column("feature_name"))).astype(np.int32), cI.get_indexer(strs(ta.column("cell_id"))).astype(np.int32),
                                cI.get_indexer(strs(ta.column("celldot_cell_id"))).astype(np.int32),
                                pd.Categorical(strs(ta.column("celldot_fate")), categories=["keep", "move", "drop"]).codes.astype(np.int8)], names="x,y,g,o,n,a")
        kb = np.rec.fromarrays([mb["x"].values.astype(np.float32), mb["y"].values.astype(np.float32), mb["gene"].values.astype(np.int32),
                                mb["old_host"].values.astype(np.int32), mb["new_host"].values.astype(np.int32),
                                pd.Categorical(mb["action"].astype(str), categories=["keep", "move", "drop"]).codes.astype(np.int8)], names="x,y,g,o,n,a")
        res["mol_all_resolved"] = bool((ka.g >= 0).all() and (ka.o >= 0).all() and ((ka.n >= 0) == (ka.a != 2)).all())
        ka.sort(); kb.sort()
        res["mol_multiset_equal"] = bool(np.array_equal(ka, kb))
        res["mol_fate_counts"] = bool(np.array_equal(np.bincount(ka.a, minlength=3), np.bincount(kb.a, minlength=3)))
        ma_n = int(ta.num_rows)
    else:
        ma_n = int(ta.num_rows)
    ma = {"n": ma_n}
    pa_ = A.uns.get("celldot", {}); pb_ = B.uns.get("spd", {})
    for k in ["n_tx", "n_cells", "n_genes", "n_types", "n_keep", "n_move", "n_drop", "drop_frac", "ambient_density"]:
        res[f"prov.{k}"] = pa_.get(k) == pb_.get(k)
    res["prov.params"] = dict(pa_.get("params", {})) == dict(pb_.get("params", {}))
    mp = json.loads((pq.read_schema(cd_mol).metadata or {}).get(b"celldot", b"{}"))
    res["prov.pair_run_id"] = mp.get("run_id") == pa_.get("run_id")

    if cd_prep and spd_prep:                         # optional: the prep artefacts (same files, minus cells_index 'row')
        for name in ["rho_tilde", "rho_corrected", "ambient"]:
            x = pd.read_parquet(cd_prep[name]); y = pd.read_parquet(spd_prep[name]); res[f"prep.{name}"] = x.equals(y)
        cx = pd.read_parquet(cd_prep["cells_index"]); cy = pd.read_parquet(spd_prep["cells_index"]).sort_values("row").drop(columns=["row"]).reset_index(drop=True)
        cy["cell_id"] = cy["cell_id"].astype(str); res["prep.cells_index"] = cx.reset_index(drop=True).equals(cy)
        mx = json.load(open(cd_prep["meta"])); my = json.load(open(spd_prep["meta"]))
        for k in ("dataset", "unassigned_value"): mx.pop(k, None); my.pop(k, None)          # CellDot-only / naming keys
        res["prep.meta"] = mx == my
    ok = all(bool(v) for v in res.values())
    w = max(len(k) for k in res)
    for k, v in res.items(): log(f"  {k:<{w}}  {'OK' if v else 'MISMATCH'}")
    log(f"cells {A.n_obs:,} | molecules {ma['n']:,} | keep/move/drop {pa_.get('n_keep'):,}/{pa_.get('n_move'):,}/{pa_.get('n_drop'):,} | "
        f"celldot run_id {pa_.get('run_id')} v{pa_.get('version')} vs spdenoise run_id {pb_.get('run_id')} v{pb_.get('version')}")
    log("RESULT: " + ("IDENTICAL" if ok else "DIFFERENT"))
    return ok, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--celldot", help="CellDot output dir (cleaned.h5ad, transcripts.parquet, …)")
    ap.add_argument("--spdenoise", help="spdenoise output dir (cleaned.h5ad, molecules.parquet, cells_index.parquet)")
    ap.add_argument("--celldot-h5ad"); ap.add_argument("--celldot-mol")
    ap.add_argument("--spd-h5ad"); ap.add_argument("--spd-mol"); ap.add_argument("--spd-cells-index")
    ap.add_argument("--prep", action="store_true", help="also compare prep artefacts found in the two dirs")
    a = ap.parse_args()
    cd_h = a.celldot_h5ad or os.path.join(a.celldot, "cleaned.h5ad"); cd_m = a.celldot_mol or os.path.join(a.celldot, "transcripts.parquet")
    sp_h = a.spd_h5ad or os.path.join(a.spdenoise, "cleaned.h5ad"); sp_m = a.spd_mol or os.path.join(a.spdenoise, "molecules.parquet")
    sp_c = a.spd_cells_index or os.path.join(a.spdenoise, "cells_index.parquet")
    prep = None
    if a.prep:
        f = lambda d: dict(rho_tilde=d + "/rho_tilde.parquet", rho_corrected=d + "/rho_tilde_corrected.parquet", ambient=d + "/ambient_profile_ag.parquet", cells_index=d + "/cells_index.parquet", meta=d + "/dataset_meta.json")
        prep = (f(a.celldot), f(a.spdenoise))
    ok, _ = compare(cd_h, cd_m, sp_h, sp_m, sp_c, *(prep or (None, None)))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
