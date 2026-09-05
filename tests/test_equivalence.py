"""End-to-end test: CellDot on a synthetic Xenium-style dataset (both id styles) — checks the cell_id output
contract, the CLI, and (when the spdenoise package is available) EXACT equality of every result with spdenoise
v0.1.1 run on the same inputs.

Run:  python tests/test_equivalence.py [work_dir]
      SPDENOISE_PKG=/path/to/spDenoise (dir containing the 'spdenoise' package; default = ../../spDenoise)
"""
import os, sys, json, subprocess, tempfile, importlib, numpy as np, pandas as pd, anndata as ad, pyarrow as pa, pyarrow.parquet as pq
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from synthetic import make_synthetic
from compare_outputs import compare
import celldot
from celldot import CellDotConfig, clean, read_provenance

SPD = os.environ.get("SPDENOISE_PKG", os.path.join(os.path.dirname(ROOT), "spDenoise"))
try:
    sys.path.insert(0, SPD); spdenoise = importlib.import_module("spdenoise"); HAVE_SPD = True
except Exception as e:
    spdenoise = None; HAVE_SPD = False; print(f"[skip] spdenoise not importable from {SPD}: {e}")


def check_format(out, fx):
    """The CellDot output contract: cleaned.h5ad indexed by cell_id; transcripts.parquet = the input table row for row,
    plus celldot_cell_id (same dtype/vocabulary as cell_id) and celldot_fate; both consistent with each other."""
    import scipy.sparse as sp
    A = ad.read_h5ad(out + "/cleaned.h5ad"); cells = pd.read_parquet(fx["outs"] + "/cells.parquet")
    assert A.obs.index.name == "cell_id" and "cell_id" not in A.obs.columns, "obs must be indexed by cell_id (no column)"
    assert A.obs_names.is_unique and set(A.obs_names) <= set(cells["cell_id"].astype(str)), "obs_names must be original cell ids"
    assert set(A.layers) == {"raw"} and "type" in A.obs, "cleaned.h5ad: X = corrected counts, layers['raw'] = before"
    tin = pq.read_table(fx["outs"] + "/transcripts.parquet"); tout = pq.read_table(out + "/transcripts.parquet")
    assert tout.num_rows == tin.num_rows, "transcripts.parquet must keep every input row"
    assert tout.column_names == tin.column_names + ["celldot_cell_id", "celldot_fate"], tout.column_names
    for c in tin.column_names: assert tout.column(c).equals(tin.column(c)), f"input column {c} altered"
    assert tout.schema.field("celldot_cell_id").type == tin.schema.field("cell_id").type, "celldot_cell_id must have the cell_id dtype"
    ft = tout.schema.field("celldot_fate").type; assert pa.types.is_dictionary(ft) and ft.value_type == pa.string()
    cid = tout.column("cell_id").to_numpy(zero_copy_only=False); ncid = tout.column("celldot_cell_id").to_numpy(zero_copy_only=False)
    fate = tout.column("celldot_fate").to_numpy(zero_copy_only=False).astype(str); gene = tout.column("feature_name").to_numpy(zero_copy_only=False).astype(str)
    sent = fx["unassigned"]; obs = set(A.obs_names); cid_s = cid.astype(str); ncid_s = ncid.astype(str)
    assert set(fate) <= {"keep", "move", "drop", "background", "not_evaluated"}
    assert ((fate == "background") == (cid == sent)).all(), "background <-> input unassigned"
    ne = fate == "not_evaluated"; assert (ncid[ne] == cid[ne]).all(), "not_evaluated rows keep their cell"
    assert (ncid[fate == "background"] == sent).all() and (ncid[fate == "drop"] == sent).all(), "background/drop -> the platform's unassigned value"
    kp = fate == "keep"; assert (ncid[kp] == cid[kp]).all() and set(cid_s[kp]) <= obs
    mv = fate == "move"; assert (ncid[mv] != cid[mv]).all() and set(ncid_s[mv]) <= obs and set(cid_s[mv]) <= obs
    ev = kp | mv | (fate == "drop"); assert set(cid_s[ev]) <= obs and set(gene[ev]) <= set(A.var_names), "evaluated molecules: labelled host, panel gene"
    qv = tout.column("qv").to_numpy(); assert (qv[ev] >= 20).all(), "evaluated molecules pass the quality cut"
    n_drop = pd.Series(cid_s[fate == "drop"]).value_counts().reindex(A.obs_names).fillna(0).astype(int).values
    assert np.array_equal(n_drop, A.obs["n_dropped"].values)
    n_in = pd.Series(ncid_s[mv]).value_counts().reindex(A.obs_names).fillna(0).astype(int).values
    assert np.array_equal(n_in, A.obs["n_moved_in"].values)
    n_out = pd.Series(cid_s[mv]).value_counts().reindex(A.obs_names).fillna(0).astype(int).values
    assert np.array_equal(n_out, A.obs["n_moved_out"].values)
    r = A.obs_names.get_indexer(ncid_s[kp | mv]); c = A.var_names.get_indexer(gene[kp | mv])
    L = sp.coo_matrix((np.ones(len(r), np.float32), (r, c)), shape=A.shape).tocsr()
    assert (L != A.X).nnz == 0, "X must equal the kept + moved molecules"
    r = A.obs_names.get_indexer(cid_s[ev]); c = A.var_names.get_indexer(gene[ev])
    R = sp.coo_matrix((np.ones(len(r), np.float32), (r, c)), shape=A.shape).tocsr()
    assert (R != A.layers["raw"]).nnz == 0, "layers['raw'] must equal the evaluated molecules by source"
    p1 = read_provenance(out + "/cleaned.h5ad"); p2 = read_provenance(out + "/transcripts.parquet")
    assert p1["run_id"] == p2["run_id"] and p1["version"] == celldot.__version__ and p2["n_tx_total"] == tout.num_rows
    for p in (p1, p2):
        assert (p["n_keep"], p["n_move"], p["n_drop"], p["n_background"], p["n_not_evaluated"]) == tuple(int((fate == k).sum()) for k in ["keep", "move", "drop", "background", "not_evaluated"]), "provenance fate counts"
    print(f"  format OK: {A.n_obs} cells | transcripts {tout.num_rows:,} rows: " + ", ".join(f"{k} {int((fate == k).sum()):,}" for k in ["keep", "move", "drop", "background", "not_evaluated"]) + f" | run_id {p1['run_id']}")
    return A


def check_viewer(out, fx):
    """The viewer bundle builds from the three viewer inputs (needs duckdb)."""
    try:
        import duckdb  # noqa
    except Exception:
        print("  viewer bundle: skipped (duckdb not installed)"); return
    from celldot.viewer.prep import build_bundle
    meta = build_bundle(out + "/cleaned.h5ad", out + "/transcripts.parquet", fx["boundaries"], out + "/viewer_bundle", log=lambda *a: None)
    p = read_provenance(out + "/cleaned.h5ad")
    assert meta["n_tx"] == p["n_keep"] + p["n_move"] + p["n_drop"] and meta["n_cells"] == ad.read_h5ad(out + "/cleaned.h5ad").n_obs
    print(f"  viewer bundle OK: {meta['n_cells_total']} cells ({meta['n_cells_polygon']} with polygons), {meta['n_tx']:,} molecules")


def main():
    work = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="celldot_test_")
    ok_all = True
    for style in ["xenium", "int"]:
        root = os.path.join(work, style); os.makedirs(root, exist_ok=True)
        fx = make_synthetic(root, seed=1 if style == "xenium" else 2, id_style=style)
        print(f"\n=== id_style={style}: {fx['n_cells']} cells, {fx['n_mol']:,} cellular + {fx['n_soup']:,} soup molecules (unassigned={fx['unassigned']!r}) ===")
        out_cd = os.path.join(root, "celldot"); os.makedirs(out_cd, exist_ok=True)
        cfg = CellDotConfig(input=fx["outs"], reference=fx["reference"], labels=fx["labels"], output=out_cd, name=f"syn_{style}")
        clean(cfg)
        check_format(out_cd, fx); check_viewer(out_cd, fx)
        # ---- CLI path must reproduce the API run exactly ----
        out_cli = os.path.join(root, "celldot_cli"); os.makedirs(out_cli, exist_ok=True)
        subprocess.run([sys.executable, "-m", "celldot", "--input", fx["outs"], "--reference", fx["reference"], "--labels", fx["labels"],
                        "--output", out_cli, "--name", f"syn_{style}"], check=True, cwd=ROOT, env={**os.environ, "PYTHONPATH": ROOT}, stdout=subprocess.DEVNULL)
        a = ad.read_h5ad(out_cd + "/cleaned.h5ad"); b = ad.read_h5ad(out_cli + "/cleaned.h5ad")
        cli_same = (a.X != b.X).nnz == 0 and (a.obs_names == b.obs_names).all() and \
                   pq.read_table(out_cd + "/transcripts.parquet").equals(pq.read_table(out_cli + "/transcripts.parquet"))
        print(f"  CLI == API: {cli_same}"); ok_all &= cli_same
        # ---- spdenoise v0.1.1 on the same inputs must give the identical result ----
        if HAVE_SPD:
            out_sp = os.path.join(root, "spdenoise"); os.makedirs(out_sp, exist_ok=True)
            scfg = spdenoise.SpdConfig(outs=fx["outs"], reference=fx["reference"], labels=fx["labels"], out=out_sp, dataset=f"syn_{style}")
            spdenoise.clean(scfg)
            f = lambda d: dict(rho_tilde=d + "/rho_tilde.parquet", rho_corrected=d + "/rho_tilde_corrected.parquet", ambient=d + "/ambient_profile_ag.parquet", cells_index=d + "/cells_index.parquet", meta=d + "/dataset_meta.json")
            ok, _ = compare(out_cd + "/cleaned.h5ad", out_cd + "/transcripts.parquet", out_sp + "/cleaned.h5ad", out_sp + "/molecules.parquet",
                            out_sp + "/cells_index.parquet", f(out_cd), f(out_sp), log=lambda *a: print(" ", *a))
            ok_all &= ok
    print("\nALL TESTS", "PASSED" if ok_all else "FAILED", f"(work dir {work})")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
