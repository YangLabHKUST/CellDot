"""Input guards: the mistakes a user can make must stop CellDot with a clear message, never a silent wrong result.

  1. --output equal to --input (would overwrite the platform's transcripts.parquet)      -> ValueError at config time
  2. Xenium 2.0-style export without scikit-image                                        -> ImportError before the scan
     (and with scikit-image present the post-dilation background path runs to the end)
  3. a normalised reference (log1p / scaled) instead of raw counts                        -> ValueError;
     layers['counts'] is used automatically; --ref-counts-layer selects a layer; a few fractional entries pass
  4. spatial labels that are not cell types of the reference                              -> ValueError naming them;
     'Unknown' style labels are allowed and reported; mismatching cell_id values          -> ValueError

Run:  python tests/test_guards.py [work_dir]      (CPU, ~1 min)
"""
import os, sys, io, json, shutil, tempfile, subprocess, contextlib, numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from synthetic import make_synthetic
from celldot import CellDotConfig
from celldot.prep import prep


def run_prep(fx, out, **kw):
    """prep() with its log captured; returns (error message or None, log text)."""
    shutil.rmtree(out, ignore_errors=True); buf = io.StringIO()
    cfg = CellDotConfig(input=fx["outs"], reference=fx["reference"], labels=fx["labels"], output=out, **kw)
    try:
        with contextlib.redirect_stdout(buf): prep(cfg)
        return None, buf.getvalue()
    except Exception as e:
        return f"{type(e).__name__}: {e}", buf.getvalue()


def expect_error(err, *needles):
    assert err is not None, "expected an error, got none"
    for n in needles: assert n.lower() in err.lower(), f"error should mention {n!r}: {err}"


def main():
    work = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="celldot_guards_")
    fx = make_synthetic(os.path.join(work, "syn"), seed=3, id_style="xenium")
    err, log = run_prep(fx, os.path.join(work, "base")); assert err is None, err
    rho0 = pd.read_parquet(os.path.join(work, "base", "rho_tilde.parquet"))
    print(f"fixture: {fx['n_cells']} cells; baseline prep OK")

    # ---- 1. output == input ----
    for out in (fx["outs"], fx["outs"] + "/", os.path.join(fx["outs"], "..", "outs")):
        try:
            CellDotConfig(input=fx["outs"], reference=fx["reference"], labels=fx["labels"], output=out); raise AssertionError(f"accepted output={out!r}")
        except ValueError as e: assert "overwrite" in str(e), e
    try:
        CellDotConfig(input=fx["outs"], reference=fx["reference"], labels=fx["labels"], output=os.path.join(work, "x"),
                      transcripts_path=os.path.join(fx["outs"], "transcripts.parquet")); raise AssertionError("accepted transcripts_path == input file")
    except ValueError as e: assert "overwrite" in str(e), e
    n0 = os.path.getsize(os.path.join(fx["outs"], "transcripts.parquet"))
    r = subprocess.run([sys.executable, "-m", "celldot", "--input", fx["outs"], "--reference", fx["reference"], "--labels", fx["labels"], "--output", fx["outs"]],
                       cwd=ROOT, env={**os.environ, "PYTHONPATH": ROOT}, capture_output=True, text=True)
    assert r.returncode != 0 and "overwrite" in r.stderr and "Traceback" not in r.stderr, r.stderr[-500:]
    assert os.path.getsize(os.path.join(fx["outs"], "transcripts.parquet")) == n0, "the input file must be untouched"
    print("1. output == input: refused (API, path variants, CLI); input file untouched")

    # ---- 2. Xenium 2.0-style export: scikit-image needed, checked before the scan; the dilation path runs ----
    outs2 = os.path.join(work, "outs2"); shutil.rmtree(outs2, ignore_errors=True); shutil.copytree(fx["outs"], outs2)
    c = pd.read_parquet(os.path.join(outs2, "cells.parquet")); c["segmentation_method"] = "Segmented by boundary stain"; c.to_parquet(os.path.join(outs2, "cells.parquet"), index=False)
    fx2 = dict(fx, outs=outs2)
    code = ("import sys; sys.modules['skimage'] = None; sys.modules['skimage.draw'] = None\n"
            f"sys.path.insert(0, {ROOT!r}); from celldot import CellDotConfig; from celldot.prep import prep\n"
            f"prep(CellDotConfig(input={outs2!r}, reference={fx['reference']!r}, labels={fx['labels']!r}, output={os.path.join(work, 'noskimage')!r}))\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0 and "scikit-image" in r.stderr and "rg 1/" not in r.stdout, (r.stdout[-300:], r.stderr[-300:])
    err, log = run_prep(fx2, os.path.join(work, "xen2")); assert err is None, err
    assert "BG_DILATE=3.0" in log and "soup absorbed" in log, log[-600:]
    print("2. Xenium 2.0-style export: ImportError before the scan without scikit-image; dilation background runs with it")

    # ---- 3. the reference must be raw counts ----
    ref = ad.read_h5ad(fx["reference"]); X = ref.X.toarray().astype(np.float64)
    ln = ad.AnnData(X=sp.csr_matrix(np.log1p(X / X.sum(1, keepdims=True) * 1e4).astype(np.float32)), obs=ref.obs.copy(), var=ref.var.copy())
    ln.write_h5ad(os.path.join(work, "ref_lognorm.h5ad"))
    err, _ = run_prep(dict(fx, reference=os.path.join(work, "ref_lognorm.h5ad")), os.path.join(work, "lognorm")); expect_error(err, "raw counts", "ref-counts-layer")
    ln.layers["counts"] = sp.csr_matrix(X.astype(np.float32)); ln.write_h5ad(os.path.join(work, "ref_lognorm_counts.h5ad"))
    err, log = run_prep(dict(fx, reference=os.path.join(work, "ref_lognorm_counts.h5ad")), os.path.join(work, "counts_layer")); assert err is None, err
    assert "layers['counts']" in log and np.allclose(pd.read_parquet(os.path.join(work, "counts_layer", "rho_tilde.parquet")).values, rho0.values)
    ln.layers["raw"] = ln.layers["counts"]; del ln.layers["counts"]; ln.write_h5ad(os.path.join(work, "ref_lognorm_raw.h5ad"))
    err, _ = run_prep(dict(fx, reference=os.path.join(work, "ref_lognorm_raw.h5ad")), os.path.join(work, "raw_layer_missing")); expect_error(err, "raw counts")
    err, log = run_prep(dict(fx, reference=os.path.join(work, "ref_lognorm_raw.h5ad")), os.path.join(work, "raw_layer"), ref_counts_layer="raw"); assert err is None, err
    assert np.allclose(pd.read_parquet(os.path.join(work, "raw_layer", "rho_tilde.parquet")).values, rho0.values)
    err, _ = run_prep(dict(fx, reference=os.path.join(work, "ref_lognorm_raw.h5ad")), os.path.join(work, "bad_layer"), ref_counts_layer="nope"); expect_error(err, "no layer")
    scaled = ad.AnnData(X=sp.csr_matrix((X - X.mean(0)).astype(np.float32)), obs=ref.obs.copy(), var=ref.var.copy()); scaled.write_h5ad(os.path.join(work, "ref_scaled.h5ad"))
    err, _ = run_prep(dict(fx, reference=os.path.join(work, "ref_scaled.h5ad")), os.path.join(work, "scaled")); expect_error(err, "negative")
    fr = ref.copy(); Xf = fr.X.toarray().astype(np.float32); rng = np.random.default_rng(0); m = (Xf > 0) & (rng.random(Xf.shape) < 0.03); Xf[m] *= 1.37   # 3% length-scaled entries (a real LuCA-style file)
    fr.X = sp.csr_matrix(Xf); fr.write_h5ad(os.path.join(work, "ref_fractional.h5ad"))
    err, log = run_prep(dict(fx, reference=os.path.join(work, "ref_fractional.h5ad")), os.path.join(work, "fractional")); assert err is None, err
    assert "not integers" in log, log[:300]
    print("3. reference counts: log-normalised and scaled refused; layers['counts'] auto-used; --ref-counts-layer honoured; few fractional entries pass with a note")

    # ---- 4. spatial labels must be reference cell types ----
    lab = pd.read_parquet(fx["labels"]); types = sorted(lab["celltype"].astype(str).unique())
    one = lab.copy(); m = one["celltype"].astype(str) == types[2]; one.loc[m, "celltype"] = types[2] + " cells"; one.to_parquet(os.path.join(work, "labels_one.parquet"), index=False)
    err, _ = run_prep(dict(fx, labels=os.path.join(work, "labels_one.parquet")), os.path.join(work, "lab_one")); expect_error(err, f"'{types[2]} cells'", f"{int(m.sum()):,} cells")
    allbad = lab.copy(); allbad["celltype"] = "type_" + allbad["celltype"].astype(str); allbad.to_parquet(os.path.join(work, "labels_all.parquet"), index=False)
    err, _ = run_prep(dict(fx, labels=os.path.join(work, "labels_all.parquet")), os.path.join(work, "lab_all")); expect_error(err, "not cell types of the reference")
    unk = lab.copy(); m = unk.index % 10 == 0; unk.loc[m, "celltype"] = "Unknown"; unk.to_parquet(os.path.join(work, "labels_unk.parquet"), index=False)
    err, log = run_prep(dict(fx, labels=os.path.join(work, "labels_unk.parquet")), os.path.join(work, "lab_unk")); assert err is None, err
    assert f"'Unknown' {int(m.sum()):,}" in log and "left uncorrected" in log, log[:600]
    ci = pd.read_parquet(os.path.join(work, "lab_unk", "cells_index.parquet")); assert not ci["cell_id"].isin(unk.loc[m, "cell_id"].astype(str)).any()
    ids = lab.copy(); ids["cell_id"] = "x" + ids["cell_id"].astype(str); ids.to_parquet(os.path.join(work, "labels_ids.parquet"), index=False)
    err, _ = run_prep(dict(fx, labels=os.path.join(work, "labels_ids.parquet")), os.path.join(work, "lab_ids")); expect_error(err, "cell_id values do not match")
    err, _ = run_prep(fx, os.path.join(work, "lab_col"), ref_label_col="cell_type"); expect_error(err, "no column 'cell_type'")
    print("4. labels: a misspelt type is refused by name and count; all-unmatched refused; 'Unknown' left uncorrected and reported; id mismatch refused; wrong --ref-label-col refused")
    print(f"\nALL GUARD TESTS PASSED (work dir {work})")


if __name__ == "__main__":
    main()
