"""End-to-end test: CellDot on a synthetic Xenium-style dataset (both id styles) — checks the cell_id output
contract, the CLI, and (when the spdenoise package is available) EXACT equality of every result with spdenoise
v0.1.1 run on the same inputs.

Run:  python tests/test_equivalence.py [work_dir]
      SPDENOISE_PKG=/path/to/spDenoise (dir containing the 'spdenoise' package; default = ../../spDenoise)
"""
import os, sys, json, subprocess, tempfile, importlib, numpy as np, pandas as pd, anndata as ad, pyarrow.parquet as pq
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
    A = ad.read_h5ad(out + "/cleaned.h5ad"); cells = pd.read_parquet(fx["outs"] + "/cells.parquet")
    assert A.obs.index.name == "cell_id" and "cell_id" not in A.obs.columns, "obs must be indexed by cell_id (no column)"
    assert A.obs_names.is_unique and set(A.obs_names) <= set(cells["cell_id"].astype(str)), "obs_names must be original cell ids"
    assert set(A.layers) == {"raw", "greedy", "celldot"}
    m = pq.read_table(out + "/molecules.parquet"); sch = {f.name: str(f.type) for f in m.schema}
    assert list(m.column_names) == ["x", "y", "gene", "old_host", "new_host", "action"], m.column_names
    for c in ["gene", "old_host", "new_host", "action"]: assert sch[c].startswith("dictionary"), (c, sch[c])
    md = m.to_pandas(); obs = set(A.obs_names)
    assert set(md["old_host"].astype(str)) <= obs, "old_host must be an obs cell_id"
    nh = md["new_host"].astype(str).values; act = md["action"].astype(str).values
    assert ((act == "drop") == (nh == "")).all(), "dropped <-> new_host == ''"
    assert set(nh[act != "drop"]) <= obs, "new_host must be an obs cell_id"
    assert ((act == "keep") == ((nh == md["old_host"].astype(str).values) & (nh != ""))).all()
    assert set(md["gene"].astype(str)) <= set(A.var_names)
    # per-cell fate counts in obs == recount from molecules
    oh = md["old_host"].astype(str).values
    nd = pd.Series(oh[act == "drop"]).value_counts().reindex(A.obs_names).fillna(0).astype(int).values
    assert np.array_equal(nd, A.obs["n_dropped"].values)
    mi = pd.Series(nh[act == "move"]).value_counts().reindex(A.obs_names).fillna(0).astype(int).values
    assert np.array_equal(mi, A.obs["n_moved_in"].values)
    # cleaned layer == molecules kept (new_host x gene)
    kept = md[act != "drop"]; r = A.obs_names.get_indexer(kept["new_host"].astype(str).values); c = A.var_names.get_indexer(kept["gene"].astype(str).values)
    import scipy.sparse as sp
    L = sp.coo_matrix((np.ones(len(kept), np.float32), (r, c)), shape=A.shape).tocsr()
    assert (L != A.layers["celldot"]).nnz == 0, "layers['celldot'] must equal the kept molecules"
    p1 = read_provenance(out + "/cleaned.h5ad"); p2 = read_provenance(out + "/molecules.parquet")
    assert p1["run_id"] == p2["run_id"] and p1["version"] == celldot.__version__
    print(f"  format OK: {A.n_obs} cells, {len(md):,} molecules, keep/move/drop {p1['n_keep']}/{p1['n_move']}/{p1['n_drop']}, run_id {p1['run_id']}")
    return A, md


def main():
    work = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="celldot_test_")
    ok_all = True
    for style in ["xenium", "int"]:
        root = os.path.join(work, style); os.makedirs(root, exist_ok=True)
        fx = make_synthetic(root, seed=1 if style == "xenium" else 2, id_style=style)
        print(f"\n=== id_style={style}: {fx['n_cells']} cells, {fx['n_mol']:,} cellular + {fx['n_soup']:,} soup molecules (unassigned={fx['unassigned']!r}) ===")
        out_cd = os.path.join(root, "celldot"); os.makedirs(out_cd, exist_ok=True)
        cfg = CellDotConfig(outs=fx["outs"], reference=fx["reference"], labels=fx["labels"], out=out_cd, dataset=f"syn_{style}")
        clean(cfg)
        check_format(out_cd, fx)
        # ---- CLI path must reproduce the API run exactly ----
        out_cli = os.path.join(root, "celldot_cli"); os.makedirs(out_cli, exist_ok=True)
        subprocess.run([sys.executable, "-m", "celldot", "--outs", fx["outs"], "--reference", fx["reference"], "--labels", fx["labels"],
                        "--out", out_cli, "--dataset", f"syn_{style}"], check=True, cwd=ROOT, env={**os.environ, "PYTHONPATH": ROOT}, stdout=subprocess.DEVNULL)
        a = ad.read_h5ad(out_cd + "/cleaned.h5ad"); b = ad.read_h5ad(out_cli + "/cleaned.h5ad")
        cli_same = (a.layers["celldot"] != b.layers["celldot"]).nnz == 0 and (a.obs_names == b.obs_names).all() and \
                   pq.read_table(out_cd + "/molecules.parquet").equals(pq.read_table(out_cli + "/molecules.parquet"))
        print(f"  CLI == API: {cli_same}"); ok_all &= cli_same
        # ---- spdenoise v0.1.1 on the same inputs must give the identical result ----
        if HAVE_SPD:
            out_sp = os.path.join(root, "spdenoise"); os.makedirs(out_sp, exist_ok=True)
            scfg = spdenoise.SpdConfig(outs=fx["outs"], reference=fx["reference"], labels=fx["labels"], out=out_sp, dataset=f"syn_{style}")
            spdenoise.clean(scfg)
            f = lambda d: dict(rho_tilde=d + "/rho_tilde.parquet", rho_corrected=d + "/rho_tilde_corrected.parquet", ambient=d + "/ambient_profile_ag.parquet", cells_index=d + "/cells_index.parquet", meta=d + "/dataset_meta.json")
            ok, _ = compare(out_cd + "/cleaned.h5ad", out_cd + "/molecules.parquet", out_sp + "/cleaned.h5ad", out_sp + "/molecules.parquet",
                            out_sp + "/cells_index.parquet", f(out_cd), f(out_sp), log=lambda *a: print(" ", *a))
            ok_all &= ok
    print("\nALL TESTS", "PASSED" if ok_all else "FAILED", f"(work dir {work})")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
