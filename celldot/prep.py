"""Stage 1 — build the CellDot data contract from raw outs + reference + labels.

Parameterised by ``CellDotConfig``. Labels are supplied externally (annotation is upstream). Writes into ``cfg.output``:
  rho_tilde.parquet            type x panel-gene row-normalised reference composition (the external prior)
  rho_tilde_corrected.parquet  RCTD-style gamma-corrected prior (USED BY THE MODEL; see below)
  cells_index.parquet          cell_id, type, x_centroid, y_centroid  (typed cells; the cell table)
  assign/shard_*.parquet       per-molecule x,y,gene(idx),cell_id(host),tx_row (row in transcripts.parquet)  (qv>=QV, panel genes, typed host)
  ambient_profile_ag.parquet   gene,a_g  (extracellular soup shrunk LAM toward panel mean)
  dataset_meta.json            N_NEG_CW,N_GENES_PANEL,f_bg,counts

Platform correction (gamma): scRNA<->Xenium platform/dropout shifts a gene's reference LEVEL (e.g. Gfap ~0 in
snRNA but huge in situ), which makes the log-rho affinity evict that gene as ambient. Per-gene factor
gamma_g = observed_spatial_pseudobulk_g / expected_g, expected_g = sum_t L_t*rho[t,g] (L_t = type-t assigned
library, from the KNOWN labels). RCTD-style, composition-corrected, contamination-robust (per-gene totals are
conserved under reassignment), per-gene so it rescales the LEVEL while preserving the cross-type CONTRAST. No
clip / no EB. Written to a SEPARATE file used ONLY by the solver; the uncorrected prior is kept for scoring.
"""
import os, json, time, glob, numpy as np, pandas as pd
import anndata as ad, scanpy as sc, scipy.sparse as sp, pyarrow.parquet as pq
from . import engine

UNLABELLED = {"unknown", "unassigned", "unlabelled", "unlabeled", "na", "nan", "none", ""}   # label values that mean 'no type': left uncorrected


def reference_counts(ref, layer=None, log=print):
    """The reference's RAW COUNTS as a CSR matrix: ``layer`` if given, else ``layers['counts']`` when present, else ``X``.
    Refuses normalised data (negative values, or mostly non-integer entries): a log-normalised reference silently
    flattens the cell-type prior, so it is an error, not a warning."""
    if layer:
        if layer not in ref.layers: raise ValueError(f"reference has no layer {layer!r} (layers: {list(ref.layers.keys())})")
        M, src = ref.layers[layer], f"layers[{layer!r}]"
    elif "counts" in ref.layers: M, src = ref.layers["counts"], "layers['counts']"
    else: M, src = ref.X, "X"
    M = M.tocsr() if sp.issparse(M) else sp.csr_matrix(np.asarray(M))
    v = M.data
    if v.size and float(v.min()) < 0:
        raise ValueError(f"reference {src} has negative values, so it is scaled data, not raw counts. CellDot needs the raw counts: "
                         f"pass --ref-counts-layer <layer> or export the counts into X.")
    frac = float(np.mean(v % 1 != 0)) if v.size else 0.0
    if frac > 0.5:
        raise ValueError(f"reference {src} does not look like raw counts ({frac:.0%} of the entries are not integers, max {float(v.max()):.2f}): "
                         f"it is probably normalised. CellDot needs the raw counts: pass --ref-counts-layer <layer> or export the counts into X.")
    log(f"reference counts: {src}" + (f" ({frac:.2%} of the entries are not integers; used as they are)" if frac > 0 else ""))
    return M


def prep(cfg):
    os.makedirs(cfg.assign_dir, exist_ok=True)
    t0 = time.time(); log = lambda *a: print(f"[prep {time.time()-t0:6.1f}s]", *a, flush=True)
    SENT = set(map(str, cfg.unassigned))                               # extracellular sentinel(s)

    # ---------- panel genes = Xenium gene-expression features ∩ reference ----------
    qh = sc.read_10x_h5(cfg.CELLFEAT); qh.var_names_make_unique()
    if "feature_types" in qh.var:
        qh = qh[:, qh.var["feature_types"].astype(str).str.contains("Gene", case=False, na=False)].copy()
    xen_panel = [str(g) for g in qh.var_names]
    ref = ad.read_h5ad(cfg.reference); ref.X = reference_counts(ref, cfg.ref_counts_layer, log)
    ref_genes = set(map(str, ref.var_names))
    panel = [g for g in xen_panel if g in ref_genes]; gidx = {g: i for i, g in enumerate(panel)}; G = len(panel)
    log(f"panel genes (Xenium ∩ ref): {G}  (Xenium {len(xen_panel)}, ref {ref.n_vars})")

    # ---------- labels -> per-cell type ----------
    lab = pd.read_parquet(cfg.labels)
    if "prob" in lab.columns: lab = lab[lab["prob"] >= cfg.min_prob]
    lcol = "celltype" if "celltype" in lab.columns else ("type" if "type" in lab.columns else lab.columns[-1])
    type_of_cid = dict(zip(lab.cell_id.astype(str), lab[lcol].astype(str)))
    cells = pd.read_parquet(cfg.CELLS)
    # Xenium 2.0 exports get the post-dilation background, which rasterises the cell polygons with scikit-image: check it
    # is installed NOW, not after the transcript scan
    _is2 = bool({"segmentation_method", "unassigned_codeword_counts", "deprecated_codeword_counts"} & set(cells.columns))  # Xenium 2.0 tight multimodal-stain seg (tight cells -> orphaned margin soup): 'segmentation_method', else the XOA-2.0 codeword columns (some 2.0 exports e.g. CRC ship these but OMIT segmentation_method); 1.0/nucleus-expansion (BC) has none
    _delta = float(getattr(cfg, "BG_DILATE", 0.0)) if _is2 else 0.0     # post-dilation background ONLY on Xenium 2.0
    if _delta > 0:
        try:
            import skimage.draw  # noqa: F401
        except ImportError as e:
            raise ImportError("this is a Xenium 2.0-style export (cells.parquet has segmentation_method / codeword columns), whose "
                              "background estimate needs scikit-image: pip install scikit-image") from e
    A_intra_all = float(np.maximum(cells["cell_area"].values.astype(float), 1.0).sum()) if "cell_area" in cells else 0.0  # ALL segmented cells
    cells["type"] = cells.cell_id.astype(str).map(type_of_cid)
    if cfg.ref_label_col not in ref.obs: raise ValueError(f"reference.obs has no column {cfg.ref_label_col!r} (columns: {list(ref.obs.columns)}); pass --ref-label-col")
    ref_types = sorted(ref.obs[cfg.ref_label_col].astype(str).unique())
    # ---- every spatial label must be one of the reference's cell types (a misspelt type would silently lose its cells) ----
    have = cells.cell_id.astype(str).isin(type_of_cid)
    if not have.any():
        raise ValueError(f"no cell of cells.parquet has a label in {cfg.labels}: the cell_id values do not match "
                         f"(labels e.g. {list(lab.cell_id.astype(str).head(3))}, cells e.g. {list(cells.cell_id.astype(str).head(3))})")
    lv = lab[lcol].astype(str); unmatched = lv[~lv.isin(ref_types)].value_counts()
    if len(unmatched):
        bad = unmatched[~unmatched.index.str.strip().str.lower().isin(UNLABELLED)]
        if len(bad):
            raise ValueError("labels that are not cell types of the reference: " + ", ".join(f"{t!r} ({n:,} cells)" for t, n in bad.items())
                             + f". The reference's {cfg.ref_label_col!r} has: {ref_types}. Use exactly these names"
                             + f" (cells to leave uncorrected may be labelled {sorted(UNLABELLED - {''})}).")
        log("cells left uncorrected (no type): " + ", ".join(f"{t!r} {n:,}" for t, n in unmatched.items()))
    if int((~have).sum()): log(f"{int((~have).sum()):,} of {len(cells):,} cells have no label -> left uncorrected")
    cells = cells[cells.type.isin(ref_types)].reset_index(drop=True)
    log(f"labelled cells: {len(cells)} | types ({cells.type.nunique()}): "
        + ", ".join(f"{t}:{n}" for t, n in cells.type.value_counts().items()))

    # ---------- rho_tilde from reference (per type, panel genes, row-normalised) ----------
    ref = ref[:, [g for g in panel]].copy(); Xr = ref.X.tocsr().astype(np.float64)
    rowsum = np.asarray(Xr.sum(1)).ravel(); rowsum[rowsum == 0] = 1.0
    Xrn = sp.diags(1.0 / rowsum) @ Xr                                  # each ref cell -> composition over panel
    rfine = ref.obs[cfg.ref_label_col].astype(str).values
    panel_mean = np.asarray(Xrn.mean(0)).ravel(); panel_mean /= max(panel_mean.sum(), 1e-12)
    spatial_counts = cells.type.value_counts()
    types_present = [t for t in ref_types
                     if int((rfine == t).sum()) >= cfg.min_ref and int(spatial_counts.get(t, 0)) >= cfg.min_spatial]
    rho = np.zeros((len(types_present), G), np.float64)
    for i, t in enumerate(types_present):
        v = np.asarray(Xrn[rfine == t].mean(0)).ravel(); rho[i] = v / max(v.sum(), 1e-12)
    pd.DataFrame(rho, index=types_present, columns=panel).to_parquet(cfg.rho)
    cells = cells[cells.type.isin(types_present)].reset_index(drop=True)
    log(f"rho_tilde {rho.shape}  types_present={len(types_present)}")

    # ---------- cells_index (typed cells) ----------
    ci = cells[["cell_id", "type", "x_centroid", "y_centroid"]].copy(); ci["cell_id"] = ci["cell_id"].astype(str)
    assert ci.cell_id.is_unique, "cells.parquet cell_id must be unique: it is the single cell identity"
    ci.to_parquet(cfg.cells_index, index=False)
    cid2pos = dict(zip(ci.cell_id.values, range(len(ci)))); log(f"cells_index: {len(ci)} cells")   # in-memory position only

    # ---------- scan transcripts -> assign shards + soup + platform pseudobulk ----------
    for f in glob.glob(cfg.assign_dir + "/shard_*.parquet"): os.remove(f)
    pf = pq.ParquetFile(cfg.TX); NRG = pf.num_row_groups
    soup = np.zeros(G, np.float64); n_soup = n_assigned = n_tx_total = 0; neg_cw = set(); unassigned_value = None
    obs_g = np.zeros(G, np.float64); cell_tot = np.zeros(len(ci), np.float64)       # assigned pseudobulk for gamma
    MPIX = 20.0; MPAD = 200.0                                                       # tissue-mask density grid (20µm)
    mx0 = float(ci.x_centroid.min() - MPAD); my0 = float(ci.y_centroid.min() - MPAD)
    mnx = int((ci.x_centroid.max() + MPAD - mx0) / MPIX) + 1; mny = int((ci.y_centroid.max() + MPAD - my0) / MPIX) + 1
    Hmask = np.zeros((mnx, mny), np.float64)                                        # all-panel-transcript density
    for rg in range(NRG):
        t = pf.read_row_group(rg, columns=["feature_name", "x_location", "y_location", "cell_id", "qv"]).to_pandas()
        fn = t.feature_name
        if len(fn) and isinstance(fn.iloc[0], (bytes, bytearray)): fn = fn.str.decode("utf-8")
        t["feature_name"] = fn.astype(str); off = n_tx_total; n_tx_total += len(t)              # off = global row of t[0]
        neg_cw |= set(x for x in t.feature_name.unique() if x.startswith("NegControlCodeword"))
        q = t[t.qv >= cfg.qv]; cq = q.cell_id.astype(str); is_panel = q.feature_name.isin(gidx)
        qp = q[is_panel]                                              # all panel transcripts → tissue-mask density
        if len(qp):
            _mx = np.clip(((qp.x_location.values - mx0) / MPIX).astype(np.int64), 0, mnx - 1)
            _my = np.clip(((qp.y_location.values - my0) / MPIX).astype(np.int64), 0, mny - 1)
            np.add.at(Hmask, (_mx, _my), 1.0)
        s = q[is_panel & cq.isin(SENT)]                               # extracellular soup
        if len(s):
            np.add.at(soup, s.feature_name.map(gidx).values, 1.0); n_soup += len(s)
            if unassigned_value is None: unassigned_value = s.cell_id.iloc[0]      # the platform's own "no cell" value, in its dtype
        a = q[is_panel & cq.isin(cid2pos)]                           # assigned to a typed cell
        if len(a):
            _g = a.feature_name.map(gidx).values.astype(np.int64); _cid = a.cell_id.astype(str)
            _h = _cid.map(cid2pos).values.astype(np.int64)
            np.add.at(obs_g, _g, 1.0); np.add.at(cell_tot, _h, 1.0)
            pd.DataFrame({"x": a.x_location.values.astype(np.float32), "y": a.y_location.values.astype(np.float32),
                          "gene": _g.astype(np.int32), "cell_id": pd.Categorical(_cid.values),   # host = original cell_id (dictionary-encoded)
                          "tx_row": (off + a.index.values).astype(np.int64)}                   # row of the molecule in transcripts.parquet
                         ).to_parquet(f"{cfg.assign_dir}/shard_{rg:04d}.parquet", index=False)
            n_assigned += len(a)
        if rg % 8 == 0 or rg == NRG - 1: log(f"  rg {rg+1}/{NRG} tx={n_tx_total:,} assigned={n_assigned:,} soup={n_soup:,}")

    # ---------- platform-effect correction (RCTD-style, MODEL-ONLY prior) ----------
    typ_row = ci.type.values
    Lt = np.array([cell_tot[typ_row == t].sum() for t in types_present])
    expg = (Lt[:, None] * rho).sum(0)
    trust = (obs_g >= cfg.gamma_nmin) & (expg >= cfg.gamma_nmin)       # genes too sparse stay at gamma=1
    gamma = np.where(trust, obs_g / np.maximum(expg, 1e-9), 1.0)
    rho_c = gamma[None, :] * rho; rho_c = rho_c / np.maximum(rho_c.sum(1, keepdims=True), 1e-12)
    pd.DataFrame(rho_c, index=types_present, columns=panel).to_parquet(cfg.rho_corrected)
    log(f"platform gamma (model-only): trusted {int(trust.sum())}/{G} | med {np.median(gamma[trust]):.2f} "
        f"max {gamma.max():.1f} (top {panel[int(gamma.argmax())]}) -> rho_tilde_corrected.parquet")

    # ---------- (optional) post-dilation background: exclude cell-MARGIN soup (tight 2.0 seg) before the ambient ----------
    A_extra_dbg = None                                                  # default => the block below is byte-identical
    if _delta > 0:                                                      # _is2 / _delta decided above, before the scan
        soup, A_extra_dbg, _dinfo = engine.dilated_background(
            cfg.TX, os.path.join(cfg.input, "cell_boundaries.parquet"), panel,
            delta=_delta, qv_min=cfg.qv, unassigned=cfg.unassigned)
        soup = soup.astype(np.float64)
        log(f"BG_DILATE={_delta}µm (Xenium 2.0 seg): soup absorbed {_dinfo['absorbed_frac']*100:.1f}% "
            f"(n_extra {_dinfo['n_soup_raw']:,}->{_dinfo['n_soup_kept']:,}) A_extra={A_extra_dbg:.3e}")
    elif getattr(cfg, "BG_DILATE", 0.0) > 0:
        log(f"BG_DILATE={cfg.BG_DILATE}µm requested but not a Xenium 2.0 export (no segmentation_method / XOA-2.0 codeword cols) -> raw soup")
    # ---------- ambient profile (DIRECT count n_extra=soup + legacy a_g for scoring) + tissue mask + meta ----------
    a_emp = soup / max(soup.sum(), 1.0); a_g = (1 - cfg.lam) * a_emp + cfg.lam * panel_mean
    pd.DataFrame({"gene": panel, "a_g": a_g, "n_extra": soup.astype(np.int64)}).to_parquet(cfg.ambient, index=False)
    # tissue-mask extracellular area (replaces the 30µm centroid grid; includes intra-tissue voids, excludes margin)
    _, _, A_tissue_mask = engine.tissue_mask_from_hist(Hmask, MPIX)
    A_intra = A_intra_all                                                       # all-cell footprint (the mask covers every cell, not only typed ones)
    A_extra = A_extra_dbg if A_extra_dbg is not None else max(A_tissue_mask - A_intra, 1.0)
    N_NEG_CW = max(len(neg_cw), 1)
    f_bg = float((cells.control_codeword_counts.sum() / N_NEG_CW) * G / max(cells.transcript_counts.sum(), 1)) \
        if "control_codeword_counts" in cells and "transcript_counts" in cells else 0.00013
    json.dump(dict(dataset=cfg.name, n_cells=int(len(ci)), n_types=int(len(types_present)), n_genes=int(G),
                   n_tx_total=int(n_tx_total), n_assigned=int(n_assigned), n_soup=int(n_soup),
                   A_tissue_mask=round(A_tissue_mask, 1), A_intra=round(A_intra, 1), A_extra=round(A_extra, 1),
                   N_NEG_CW=int(N_NEG_CW), N_GENES_PANEL=int(G), f_bg=round(f_bg, 6), types=types_present,
                   unassigned_value=(unassigned_value.item() if hasattr(unassigned_value, "item") else unassigned_value)),
              open(cfg.meta, "w"), indent=1)
    log(f"DONE prep  cells={len(ci)} genes={G} types={len(types_present)} assigned_tx={n_assigned:,} "
        f"soup_top={panel[int(soup.argmax())]} A_extra(mask)={A_extra:.3e} f_bg={f_bg:.6f}")
