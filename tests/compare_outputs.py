"""Compare a CellDot run against a spdenoise (v0.1.1) run on the SAME inputs — the two must agree exactly.

spdenoise names cells by integer row (cells_index.parquet 'row' -> cell_id; molecules old/new_host = row,
-1 = dropped; molecules gene = var index). CellDot names them by the original cell_id (obs_names; molecules
old/new_host = cell_id, "" = dropped; gene = name). This maps the spdenoise output onto the cell_id scheme
and checks every layer, obs column, molecule fate and provenance count.

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
    for la, lb in [("raw", "raw"), ("greedy", "greedy"), ("celldot", "spdenoise")]:
        res[f"layer_{la}=={lb}"] = _same_sparse(A.layers[la], B.layers[lb])
    res["X==raw"] = _same_sparse(A.X, A.layers["raw"])
    for c in ["x_centroid", "y_centroid", "mu_bg", "n_dropped", "n_moved_out", "n_moved_in", "drop_frac"]:
        res[f"obs.{c}"] = bool(np.array_equal(A.obs[c].values, B.obs[c].values))
    res["obs.type"] = bool((A.obs["type"].astype(str).values == B.obs["type"].astype(str).values).all())
    res["var.lambda0"] = bool(np.array_equal(A.var["lambda0"].values, B.var["lambda0"].values))

    # molecules: compare through dictionary CODES (no per-row string materialisation; the BC table has ~40M rows)
    ma = pq.read_table(cd_mol, read_dictionary=["gene", "old_host", "new_host", "action"]).to_pandas()
    mb = pq.read_table(spd_mol, read_dictionary=["action"]).to_pandas()
    res["mol_nrows"] = len(ma) == len(mb)
    if res["mol_nrows"]:
        res["mol_xy"] = bool(np.array_equal(ma["x"].values, mb["x"].values) and np.array_equal(ma["y"].values, mb["y"].values))
        def codes(col): return col.cat.codes.values.astype(np.int64), pd.Index(col.cat.categories.astype(str))
        gc, gcat = codes(ma["gene"]); g_map = gcat.get_indexer(np.asarray(list(B.var_names), dtype=object))   # spd gene idx -> celldot code
        res["mol_gene"] = bool((g_map >= 0).all() and np.array_equal(gc, g_map[mb["gene"].values.astype(np.int64)]))
        oc, ocat = codes(ma["old_host"]); nc, ncat = codes(ma["new_host"])
        o_map = ocat.get_indexer(cid_spd); n_map = ncat.get_indexer(cid_spd); n_empty = ncat.get_loc("") if "" in ncat else -2
        res["mol_old_host"] = bool((o_map >= 0).all() and np.array_equal(oc, o_map[mb["old_host"].values.astype(np.int64)]))
        nh = mb["new_host"].values.astype(np.int64)
        res["mol_new_host"] = bool((n_map >= 0).all() and n_empty >= 0 and np.array_equal(nc, np.where(nh >= 0, n_map[np.clip(nh, 0, None)], n_empty)))
        ac, acat = codes(ma["action"]); a_b = mb["action"].astype("category"); a_map = acat.get_indexer(a_b.cat.categories.astype(str))
        res["mol_action"] = bool((a_map >= 0).all() and np.array_equal(ac, a_map[a_b.cat.codes.values.astype(np.int64)]))
        res["mol_drop<->empty_new_host"] = bool(np.array_equal(ac == acat.get_loc("drop"), nc == n_empty))
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
        mx = json.load(open(cd_prep["meta"])); my = json.load(open(spd_prep["meta"])); mx.pop("dataset", None); my.pop("dataset", None); res["prep.meta"] = mx == my
    ok = all(bool(v) for v in res.values())
    w = max(len(k) for k in res)
    for k, v in res.items(): log(f"  {k:<{w}}  {'OK' if v else 'MISMATCH'}")
    log(f"cells {A.n_obs:,} | molecules {len(ma):,} | keep/move/drop {pa_.get('n_keep'):,}/{pa_.get('n_move'):,}/{pa_.get('n_drop'):,} | "
        f"celldot run_id {pa_.get('run_id')} v{pa_.get('version')} vs spdenoise run_id {pb_.get('run_id')} v{pb_.get('version')}")
    log("RESULT: " + ("IDENTICAL" if ok else "DIFFERENT"))
    return ok, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--celldot", help="CellDot output dir (cleaned.h5ad, molecules.parquet, …)")
    ap.add_argument("--spdenoise", help="spdenoise output dir (cleaned.h5ad, molecules.parquet, cells_index.parquet)")
    ap.add_argument("--celldot-h5ad"); ap.add_argument("--celldot-mol")
    ap.add_argument("--spd-h5ad"); ap.add_argument("--spd-mol"); ap.add_argument("--spd-cells-index")
    ap.add_argument("--prep", action="store_true", help="also compare prep artefacts found in the two dirs")
    a = ap.parse_args()
    cd_h = a.celldot_h5ad or os.path.join(a.celldot, "cleaned.h5ad"); cd_m = a.celldot_mol or os.path.join(a.celldot, "molecules.parquet")
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
