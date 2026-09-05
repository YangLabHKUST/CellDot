"""celldot-annotate: cell-type labels for the spatial cells by joint scANVI label transfer (the recipe used in the paper).

    celldot-annotate --input <platform output folder> --reference ref.h5ad --output labels.parquet
                     [--ref-label-col celltype] [--hvg 4000] [--ref-batch-col sample] [--cpu]

Reads the per-cell counts the platform exports (cell_feature_matrix.h5), trains one scVI model on the reference and
the spatial cells together (raw counts; batch = reference / spatial), then scANVI with the reference labels observed,
and writes labels.parquet with cell_id, celltype and prob (the posterior probability of the chosen type).
Needs scvi-tools (`pip install "celldot[annotate]"`). Any other annotation works as well, as long as it produces
the same table with the reference's cell-type names.
"""
import argparse, os, warnings
import numpy as np, pandas as pd


def _as_counts(a, counts_layer=None):
    a = a.copy()
    if counts_layer and counts_layer in a.layers: a.X = a.layers[counts_layer].copy()
    X = a.X; frac = (X.data % 1 != 0).mean() if hasattr(X, "data") else float(np.mean(X % 1 != 0))
    if frac > 1e-6: warnings.warn(f"{a.shape}: .X does not look like integer counts (non-integer fraction {frac:.3f})")
    return a


def annotate_scanvi_joint(query, reference, label_key="celltype", ref_batch_key=None, query_counts_layer=None, ref_counts_layer=None,
                          use_hvg=False, n_hvg=4000, n_latent=30, n_layers=2, gene_likelihood="nb", max_epochs_scvi=None,
                          max_epochs_scanvi=25, n_samples_per_label=100, seed=0, gpu=True, log=print):
    """Joint (transductive) scANVI: reference + query co-embedded; returns a DataFrame (pred, prob) indexed like query."""
    import anndata as ad, scanpy as sc, scvi, torch
    try: torch.set_float32_matmul_precision("high")
    except Exception: pass
    scvi.settings.seed = seed; acc = "gpu" if (gpu and torch.cuda.is_available()) else "cpu"
    if label_key not in reference.obs: raise ValueError(f"reference.obs has no column {label_key!r}")
    q = _as_counts(query, query_counts_layer); r = _as_counts(reference, ref_counts_layer)
    q.var_names_make_unique(); r.var_names_make_unique()
    common = [g for g in r.var_names if g in set(q.var_names)]
    if len(common) < 100: raise ValueError(f"only {len(common)} genes shared by the panel and the reference")
    r = r[:, common].copy(); q = q[:, common].copy()
    if use_hvg and len(common) > n_hvg:
        sc.pp.highly_variable_genes(r, n_top_genes=n_hvg, flavor="seurat_v3")
        hvg = list(r.var_names[r.var["highly_variable"]]); r = r[:, hvg].copy(); q = q[:, hvg].copy()
    log(f"  genes used: {r.n_vars} (shared {len(common)}, hvg {'on' if use_hvg else 'off'}) | device {acc}")
    LBL, BATCH, TECH = "_scanvi_label", "_scanvi_batch", "_scanvi_tech"
    r.obs[LBL] = r.obs[label_key].astype(str).values; q.obs[LBL] = "Unknown"
    r.obs[BATCH] = ("ref:" + r.obs[ref_batch_key].astype(str).values) if ref_batch_key else "reference"; q.obs[BATCH] = "query"
    r.obs[TECH] = "reference"; q.obs[TECH] = "query"
    r.obs_names = ["ref-" + b for b in r.obs_names]
    comb = ad.concat([r, q], join="inner"); comb.layers["counts"] = comb.X.copy()
    log(f"  combined: {comb.n_obs:,} cells ({int((comb.obs[TECH] == 'query').sum()):,} spatial)")
    scvi.model.SCVI.setup_anndata(comb, layer="counts", batch_key=BATCH)
    m = scvi.model.SCVI(comb, n_latent=n_latent, n_layers=n_layers, gene_likelihood=gene_likelihood)
    m.train(max_epochs=max_epochs_scvi, early_stopping=True, accelerator=acc, devices=1, batch_size=1024, enable_progress_bar=False)
    lvae = scvi.model.SCANVI.from_scvi_model(m, unlabeled_category="Unknown", labels_key=LBL)
    lvae.train(max_epochs=max_epochs_scanvi, n_samples_per_label=n_samples_per_label, accelerator=acc, devices=1, batch_size=1024, enable_progress_bar=False)
    is_q = (comb.obs[TECH] == "query").values
    soft = lvae.predict(soft=True); hard = np.asarray(lvae.predict())
    out = pd.DataFrame({"pred": hard[is_q], "prob": soft.values[is_q].max(1)}); out.index = query.obs_names
    return out


def read_spatial_counts(input_dir, cellfeat_name="cell_feature_matrix.h5"):
    """The platform's per-cell count matrix (10x HDF5) as AnnData with obs_names = cell_id, gene-expression features only."""
    import scanpy as sc
    a = sc.read_10x_h5(os.path.join(input_dir, cellfeat_name), gex_only=False); a.var_names_make_unique()
    if "feature_types" in a.var: a = a[:, a.var["feature_types"].astype(str) == "Gene Expression"].copy()
    return a


def main():
    ap = argparse.ArgumentParser(prog="celldot-annotate", description="cell-type labels for the spatial cells by scANVI label transfer from the reference")
    ap.add_argument("--input", required=True, help="the platform's output folder (cell_feature_matrix.h5 is read)")
    ap.add_argument("--reference", required=True, help="single-cell reference .h5ad (raw counts)")
    ap.add_argument("--output", required=True, help="labels.parquet to write (cell_id, celltype, prob)")
    ap.add_argument("--ref-label-col", default="celltype", help="column of reference.obs with the cell type (default: celltype)")
    ap.add_argument("--ref-batch-col", help="column of reference.obs with a batch / sample id, corrected alongside the technology")
    ap.add_argument("--hvg", type=int, default=0, help="train on this many highly variable genes instead of all shared genes (recommended 4000 for 5K panels)")
    ap.add_argument("--epochs-scanvi", type=int, default=25); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true", help="do not use the GPU")
    a = ap.parse_args()
    import anndata as ad
    q = read_spatial_counts(a.input); r = ad.read_h5ad(a.reference)
    print(f"spatial cells {q.n_obs:,} x {q.n_vars:,} genes | reference {r.n_obs:,} cells, {r.obs[a.ref_label_col].nunique()} types", flush=True)
    pred = annotate_scanvi_joint(q, r, label_key=a.ref_label_col, ref_batch_key=a.ref_batch_col, use_hvg=a.hvg > 0, n_hvg=a.hvg,
                                 max_epochs_scanvi=a.epochs_scanvi, seed=a.seed, gpu=not a.cpu)
    out = pd.DataFrame({"cell_id": pred.index.astype(str), "celltype": pred["pred"].astype(str).values, "prob": pred["prob"].astype(np.float32).values})
    out.to_parquet(a.output, index=False)
    print(f"wrote {a.output}: {len(out):,} cells\n" + out["celltype"].value_counts().to_string(), flush=True)


if __name__ == "__main__":
    main()
