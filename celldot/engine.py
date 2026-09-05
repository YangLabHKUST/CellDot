"""CellDot engine — sparse unbalanced entropic optimal transport (Sinkhorn) with a reference prior, NB
over-dispersed expression capacities and a per-cell physical background capacity, plus the sliding-window
background decode (``neighborhood_decode``) and the background/ambient estimators used by ``prep``/``run``.

The code is identical to spdenoise v0.1.1 ``engine.py`` (the validated engine); only this header differs.
Cells are addressed by in-memory position throughout; the original cell_id is attached by ``run``.
"""
import numpy as np, pandas as pd, torch
try:
    import shapely
    from shapely import Polygon, Point, points as shp_points, distance as shp_distance
    HAVE_SHAPELY = True
except Exception as e:
    HAVE_SHAPELY = False; _SHP_ERR = str(e)


def estimate_fano(raw, ctype, NT, G, clip=(1.0, 50.0), min_cells=30):
    """Within-type Fano factor F[t,g] = Var/Mean of raw counts over type-t cells (the over-dispersion)."""
    F = np.ones((NT, G), np.float32)
    for t in range(NT):
        ix = np.where(ctype == t)[0]
        if len(ix) < min_cells: continue
        sub = raw[ix]; m = np.asarray(sub.mean(0)).ravel()
        m2 = np.asarray(sub.multiply(sub).mean(0)).ravel(); var = np.maximum(m2 - m * m, 0)
        F[t] = np.where(m > 1e-6, var / np.maximum(m, 1e-6), 1.0)
    return np.clip(F, clip[0], clip[1])


def estimate_ambient_profile(transcripts_path, genes, panel_mean, lam=0.1, qv_min=20.0, unassigned="UNASSIGNED"):
    """④ Gene-specific ambient (soup) profile a_g from EXTRACELLULAR transcripts (cell_id==UNASSIGNED), shrunk
    toward the panel-mean reference: a_g = (1-lam)*emp_g + lam*panel_mean_g, normalised to sum 1. Only the SHAPE
    is learned here; the drop RATE stays the neg-control F_BG (the solver applies a_g mean-preserving). Predicate
    is pushed into the parquet scan so only the unassigned, is_gene, qv>=qv_min rows are read. Returns (a_g, info)."""
    import pyarrow.parquet as pq, pyarrow.dataset as pds, pyarrow.compute as pc
    name2idx = {g: i for i, g in enumerate(genes)}; G = len(genes)
    _md = pq.ParquetFile(transcripts_path).metadata; n_total = sum(_md.row_group(i).num_rows for i in range(_md.num_row_groups))   # footer num_rows is wrong in some exports
    filt = (pc.field("cell_id") == unassigned) & pc.field("is_gene")
    if qv_min is not None: filt = filt & (pc.field("qv") >= float(qv_min))
    tbl = pds.dataset(transcripts_path, format="parquet").scanner(columns=["feature_name"], filter=filt).to_table()
    fn = pd.Series(tbl.column("feature_name").to_pandas())
    idv = fn.map(name2idx).to_numpy(); ok = ~pd.isna(idv)
    n_extra = np.bincount(idv[ok].astype(np.int64), minlength=G).astype(np.float64)
    pm = np.asarray(panel_mean, np.float64); pm = pm / max(pm.sum(), 1e-12)
    emp = n_extra / max(n_extra.sum(), 1.0)
    a = (1.0 - lam) * emp + lam * pm; a = a / a.sum()
    order = np.argsort(-a)
    info = dict(n_total=int(n_total), n_extracellular=int(len(fn)), n_extracellular_in_panel=int(ok.sum()),
                frac_extracellular=round(len(fn) / max(n_total, 1), 5), lam=float(lam), qv_min=qv_min,
                corr_emp_panelmean=round(float(np.corrcoef(emp, pm)[0, 1]), 4),
                top=[(genes[i], int(n_extra[i]), round(float(a[i]), 5)) for i in order[:15]])
    return a.astype(np.float32), info


def tissue_mask_from_hist(H, pix, sigma_um=40.0, keep_frac=0.999, close_um=40.0):
    """Tissue mask from a 2D transcript-density histogram H[nx,ny] (counts per `pix`-µm cell).
    Gaussian-smooth → threshold to the SMALLEST pixel set holding `keep_frac` of total density → fill enclosed
    voids → light morphological close. Tissue = where the molecules are, so it INCLUDES ambient-filled intra-tissue
    voids (their scattered transcripts give nonzero density) and EXCLUDES the empty slide margin (~0 density). This
    replaces the 30µm centroid-occupancy A_tissue, which dropped any void with no cell centroid (the Bug-1 cause).
    Returns (mask[nx,ny] bool, D smoothed-density, area_um2)."""
    from scipy.ndimage import gaussian_filter, binary_fill_holes, binary_closing
    D = gaussian_filter(H.astype(np.float64), sigma=max(sigma_um / pix, 0.5))
    fl = D.ravel(); order = np.argsort(fl)[::-1]; cs = np.cumsum(fl[order]); tot = max(cs[-1], 1e-12)
    cut = fl[order][min(int(np.searchsorted(cs, keep_frac * tot)), len(order) - 1)]
    mask = D >= max(cut, 1e-12)
    mask = binary_fill_holes(mask)                                     # internal (enclosed) voids -> tissue
    if close_um > 0:
        mask = binary_closing(mask, iterations=max(int(round(close_um / pix)), 1)); mask = binary_fill_holes(mask)
    return mask, D, float(mask.sum()) * pix * pix


def background_params(n_extra, A_extra, eps=1e-3):
    """Direct-count physical background (clean design, redesign.html §4) — the SINGLE source of (λ⁰, p_bg), called by
    both run.py and the experiment driver so they match EXACTLY. From the raw per-gene extracellular count
    n_extra_g (UNASSIGNED, panel) and the tissue-mask extracellular area A_extra:
       p_bg[g] = (n_extra_g + eps) / (Σ n_extra + eps·G)   — SHAPE (which genes drop); eps floors p_bg>0 so a
                                                             molecule with no in-range candidate is dropped.
       λ⁰[g]   = n_extra_g / A_extra                        — DENSITY [tx/µm²] (how many).
    Returns (lambda0 float32, p_bg float32)."""
    n = np.asarray(n_extra, np.float64); Gn = len(n)
    p_bg = ((n + eps) / (n.sum() + eps * Gn)).astype(np.float32)
    lambda0 = (n / max(float(A_extra), 1.0)).astype(np.float32)
    return lambda0, p_bg


def dilated_background(tx_path, cell_boundaries_path, genes, delta=3.0, pix=2.0, pad=100.0,
                       qv_min=20.0, unassigned=("UNASSIGNED", "-1"), sigma_um=40.0, keep_frac=0.999, close_um=40.0):
    """Post-dilation background (redesign.html §5) — for TIGHT 2.0 interior/boundary-stain segmentation, whose
    cells barely exceed the nucleus, the 'extracellular soup' is contaminated by orphaned cell-MARGIN transcripts
    (real cellular molecules just outside the membrane). They make the soup whole-cell-like, inflating BOTH λ⁰ (the
    budget) and the cell-like genes in p_bg. This excludes UNASSIGNED molecules within `delta` µm of a cell before
    estimating the ambient:
       rasterize the cell-boundary polygons on a `pix`-µm grid -> dilate by `delta` (exact EDT) -> n_extra'_g =
       #UNASSIGNED of gene g OUTSIDE the dilated cells; A_extra' = tissue_mask_area − dilated_cell_area.
    delta=0 reproduces the raw background (sanity). ONE streaming pass over transcripts.  Returns
    (n_extra' [G] int64, A_extra' float, info dict).  Used by prep when cfg.BG_DILATE>0; default path untouched."""
    import pyarrow.parquet as pq
    from skimage.draw import polygon as skpoly
    from scipy.ndimage import distance_transform_edt
    name2idx = {g: i for i, g in enumerate(genes)}; G = len(genes); SENT = set(map(str, unassigned))
    bdf = pd.read_parquet(cell_boundaries_path, columns=["cell_id", "vertex_x", "vertex_y"])
    vx = bdf["vertex_x"].values.astype(np.float64); vy = bdf["vertex_y"].values.astype(np.float64)
    x0 = float(vx.min() - pad); y0 = float(vy.min() - pad)
    nx = int((vx.max() + pad - x0) / pix) + 1; ny = int((vy.max() + pad - y0) / pix) + 1
    # rasterize REAL cell polygons (contiguous per-cell vertex runs), then exact Δµm dilation
    vr = (vx - x0) / pix; vc = (vy - y0) / pix; cid = bdf["cell_id"].values
    od = np.argsort(cid, kind="stable"); vr = vr[od]; vc = vc[od]; cid = cid[od]
    seg = np.flatnonzero(np.r_[True, cid[1:] != cid[:-1], True])
    cellmask = np.zeros((nx, ny), bool)
    for a, b in zip(seg[:-1], seg[1:]):
        rr, cc = skpoly(vr[a:b], vc[a:b], shape=(nx, ny)); cellmask[rr, cc] = True
    dilmask = (cellmask | (distance_transform_edt(~cellmask) * pix <= delta)) if delta > 0 else cellmask
    # stream transcripts: tissue-mask density H (all panel) + per-gene UNASSIGNED OUTSIDE the dilated cells
    n_extra = np.zeros(G, np.float64); H = np.zeros((nx, ny), np.float64); n_soup_all = 0
    pf = pq.ParquetFile(tx_path)
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=["feature_name", "x_location", "y_location", "cell_id", "qv"]).to_pandas()
        fn = t["feature_name"]
        if len(fn) and isinstance(fn.iloc[0], (bytes, bytearray)): fn = fn.str.decode("utf-8")
        t["feature_name"] = fn.astype(str); q = t[t["qv"] >= qv_min]
        gi = q["feature_name"].map(name2idx); ok = gi.notna(); qq = q[ok]
        if not len(qq): continue
        gv = gi[ok].values.astype(np.int64)
        ix = np.clip(((qq["x_location"].values - x0) / pix).astype(np.int64), 0, nx - 1)
        iy = np.clip(((qq["y_location"].values - y0) / pix).astype(np.int64), 0, ny - 1)
        np.add.at(H, (ix, iy), 1.0)
        isun = qq["cell_id"].astype(str).isin(SENT).values
        keep = isun & (~dilmask[ix, iy])                                  # soup molecule OUTSIDE the dilated cells
        if keep.any(): np.add.at(n_extra, gv[keep], 1.0)
        n_soup_all += int(isun.sum())
    mask, _, A_tissue = tissue_mask_from_hist(H, pix, sigma_um=sigma_um, keep_frac=keep_frac, close_um=close_um)
    A_extra = max(float((mask & ~dilmask).sum()) * pix * pix, 1.0)
    info = dict(delta=float(delta), pix=float(pix), n_soup_raw=int(n_soup_all), n_soup_kept=int(n_extra.sum()),
                absorbed_frac=round(1 - n_extra.sum() / max(n_soup_all, 1), 4), A_extra=round(A_extra, 1),
                A_extra_orig=round(float((mask & ~cellmask).sum()) * pix * pix, 1),
                A_cells=round(float(cellmask.sum()) * pix * pix, 1), A_dilated=round(float(dilmask.sum()) * pix * pix, 1),
                A_tissue=round(float(A_tissue), 1))
    return n_extra.astype(np.int64), float(A_extra), info


def load_polys(XO, cid2row, NCELL, cxy, cr):
    """All cell polygons indexed by cell row (fallback to centroid circle for missing/invalid)."""
    assert HAVE_SHAPELY, "shapely unavailable"
    bdf = pd.read_parquet(XO + "/cell_boundaries.parquet", columns=["cell_id", "vertex_x", "vertex_y"])
    poly = np.empty(NCELL, object)
    for cid, grp in bdf.groupby("cell_id", sort=False):
        r = cid2row.get(cid)
        if r is None: continue
        v = grp[["vertex_x", "vertex_y"]].values
        if len(v) >= 3:
            try:
                p = Polygon(v); poly[r] = p if p.is_valid else p.buffer(0)
            except Exception:
                poly[r] = None
    for r in range(NCELL):
        if poly[r] is None or (hasattr(poly[r], "is_empty") and poly[r].is_empty):
            poly[r] = Point(cxy[r, 0], cxy[r, 1]).buffer(max(float(cr[r]), 0.5))
    return poly


def tile_distance(xp, yp, idx, dist_centroid, cr, poly_arr, mode):
    """(M,K) molecule->candidate distance: 'boundary' = membrane (shapely), else centroid-minus-radius."""
    if mode == "boundary" and poly_arr is not None:
        pg = shp_points(np.asarray(xp, float), np.asarray(yp, float)); M, K = idx.shape
        d = np.zeros((M, K), np.float32)
        for k in range(K):
            d[:, k] = shp_distance(pg, poly_arr[idx[:, k]])
        return d
    return np.clip(dist_centroid - cr[idx], 0, None).astype(np.float32)


def sinkhorn_solve(aff, within, candU, gp, ctypeU, NrawU, RHOt, Ften, G, dev,
                   lambda0=None, host=None, area_host=None, budget="nb", pw=0.3, drop=True, Z=2.0, eps=1.0,
                   niter=200, kappa=1.0, pw_bg=1.0, p_bg=None, rho_cell=None, drop_decode="argmax", host_slot=None,
                   bg_scale="percell", host_ctype=None, bg_alpha=0.12, bg_soft=0.0):
    """Per-gene unbalanced Sinkhorn with a PER-CELL PHYSICAL background — the ONLY background mechanism.
    (The legacy flat-F_BG / shaped-soup single global drop bin was REMOVED 2026-06-17; see git history + the
    redesign.html / findings.html docs for why it was panel-size-dependent.)

    The transport assigns each molecule over its K candidate cells (NATIVE) plus one background sink keyed by its
    HOST cell. Two column budgets, projected with DIFFERENT firmness:
      NATIVE column (k,g)  — SOFT.  target mu_hi = N_k*rho~[t_k,g] + Z*sqrt(Fano*mu) (over-dispersed NB band);
        vb = min(1, (mu_hi/colload)^pw), pw small (0.3) so within-type heterogeneity is preserved (over-projecting
        the native budget evicts real above-mean signal).
      BACKGROUND column — FIRM.  PHYSICAL capacity, kernel K_bg[m] = p_bg[gene] (the ambient spectrum, sum-1 over
        genes -> same ~1/G footing as rho~, so the drop decision is panel-size-INVARIANT). v_bg = min(1,(mu_bg/
        bgload)^pw_bg), pw_bg ~ 1 (firm) because mu_bg is a physical measurement we trust. Two granularities:
          bg_scale='percell' (default): column = (host cell, gene), capacity mu_bg[k,g] = kappa*lambda0[g]*area_k.
            A cell sheds AT MOST mu_bg of gene g (per-cell deletion-proof). BUT lambda0*small-cell-area is sub-unit
            for most (cell,gene) => the firm cap is split across n molecules (the split bug; see drop_decode='leak').
          bg_scale='metacell': ONE column per gene per TILE, capacity mu_bg[g] = kappa*lambda0[g]*A_meta where
            A_meta = SUM of the tile's unique cell areas (all cells = one 'meta-cell'). == the SUM of the per-cell
            caps (total conserved), but applies the density at a scale where lambda0*area is a reliable COUNT, and
            lets the tile-wide affinity competition (not the noisy per-cell area) decide which molecules drop.
            Per-tile deletion-proof (tile sheds <= its physical soup of gene g).
          bg_scale='metacell_typed': metacell, but PARTITIONED by host cell-type to stop cross-type spill in mixed
            tiles (a high-lambda0 gene that is a genuine marker of a LOW-rho~ type was over-cleaned where contaminant
            cells inflate A_meta and the per-gene leak floor spills the drop into the genuine source -> tile-square
            borders). Per gene, a type is an 'expresser' iff rho~[t,g] >= bg_alpha*max_t rho~[t,g]: each expresser
            type gets its OWN sink (capacity kappa*lambda0[g]*A_t, A_t = its cells' area), and ALL non-expresser
            types are POOLED into one sink (capacity kappa*lambda0[g]*A_pool) so the pooled count stays reliable. The
            leak floor is then per SINK (not per gene), so a genuine type is only drained by its own floor(lambda0*
            A_t). Sum of sink capacities == kappa*lambda0[g]*A_meta (total drop conserved vs metacell). bg_soft>0
            adds a soft Kbg*=(1-rho~/rho~max)^bg_soft genuine-marker repel. Needs host_ctype (M,) = host type index.
          bg_scale='neighborhood': Sinkhorn background == 'percell' (per-cell physical cap, tile-free Pd); the DROP
            COUNT is deferred to a SLIDING fixed-RADIUS window (engine.neighborhood_decode, called from run.py) so the
            per-tile leak floor's 500um grid blocks disappear. Returns (hard, margin, eligible) per molecule instead of
            a hard decode; run.py's Pass-2 sheds floor(r*n) per cell, r=min(1, kappa*lambda0[g]*A_window / E_window).
      Drop(m) iff  u*K_bg*v_bg  >  max_k u*Km*vb.   [drop_decode='argmax', the per-molecule hard decode]
      == the reference-prior model + a per-cell 'background neighbour' candidate + a (firm) cap on its count.

    drop_decode='leak' (the cell x gene MATRIX-faithful decode): the per-molecule arg-max above SPLITS a shared
      sub-unit background cap across the n_mol molecules of a (cell,gene) column (each gets ~mu_bg/n_mol), so a
      column that should shed 1 transcript keeps ALL when crowded (the 'split bug'). 'leak' fixes the COUNT: after
      the arg-max, among KEPT molecules (won by the HOST slot -> host_slot needed; MOVES are excluded, they found a
      genuine home) that still prefer background absent the cap (margin = u*K_bg - best native > 0), it sheds per
      GENE an extra floor(sum of their realized background mass Pd) of them, ranked by that margin (the uncapped
      desire — strips the v_bg saturation factor so crowded/small-cell columns aren't under-ranked). Summing Pd at
      the GENE level BEFORE flooring recovers the sub-unit tail that a per-column round() would zero. Deletion-proof
      relaxes from per-cell to per-gene (tissue sheds <= gene's physical soup). host_slot (M,) = the candidate slot
      whose cell == the molecule's host (-1 if host not among the K candidates); required for drop_decode='leak'.

    Inputs (drop=True requires these; per-molecule / per-gene tensors already on dev):
      lambda0 (G,)   free-ambient areal density [tx/um^2] = n_extra_g / A_extra  (DIRECT count / tissue-mask area;
                     see engine.background_params / redesign.html §4)
      host    (M,)   host cell id of each molecule (the footprint owner the ambient is attributed to)
      area_host (M,) 2D segmentation area of each molecule's host cell [um^2]
      kappa          capture-efficiency scalar (1.0 = the literal physical lambda0*area estimate)
      p_bg (G,)      background kernel = (n_extra+eps)/(Σn_extra+eps·G); default falls back to lambda0/lambda0.sum()
    drop=False => pure move-only (no background; greedy/relocation reference).
    Returns hard winner slot per molecule (-1 = dropped to ambient)."""
    M, K = aff.shape
    Km = torch.exp(aff / eps) * within
    flat = (candU.to(torch.int64) * G + gp[:, None].to(torch.int64)).reshape(-1)   # int64 key: avoid U*G overflow
    ub, inv = torch.unique(flat, return_inverse=True); invK = inv.view(M, K)
    jb = ub // G; gb = ub % G; ct = ctypeU[jb].clamp_min(0)
    rho_b = rho_cell[jb, gb] if rho_cell is not None else RHOt[ct, gb]    # v3: per-cell soft-type mixture reference
    mu_b = (NrawU[jb] * rho_b).clamp_min(1e-8)
    target = mu_b + Z * torch.sqrt(Ften[ct, gb] * mu_b) if budget == "nb" else mu_b
    u = torch.ones(M, device=dev); vb = torch.ones(ub.numel(), device=dev)

    if drop:                                                             # --- physical background column(s) ---
        assert lambda0 is not None and host is not None and area_host is not None, \
            "drop=True needs lambda0, host, area_host (the per-cell background; legacy F_BG bin was removed)"
        if p_bg is None: p_bg = (lambda0 + 1e-30) / (lambda0.sum() + 1e-30 * G)   # floor>0 so all-zero lambda0 still drops no-candidate molecules
        Kbg = p_bg[gp].clamp_min(0.0)                                    # ambient-spectrum kernel (sum-1 over genes)
        if bg_scale == "metacell":                                      # TILE meta-cell: all cells -> one big cell of area A_meta;
            uh, inv_h = torch.unique(host, return_inverse=True)         #   budget pooled per GENE over the tile (density at a valid scale)
            cnt = torch.bincount(inv_h, minlength=uh.numel()).to(area_host.dtype)
            A_meta = (area_host / cnt[inv_h]).sum()                     # = SUM of UNIQUE host-cell areas (each cell counted once)
            bg_flat = gp.to(torch.int64)                               # background column key = gene only (one sink per gene per tile)
            bg_ub, bg_inv = torch.unique(bg_flat, return_inverse=True)  # bg_ub = unique genes present in the tile
            mu_bg_col = (kappa * lambda0[bg_ub] * A_meta).clamp_min(1e-12)   # mu_bg[g] = kappa * lambda0[g] * A_meta  (== sum of the per-cell caps)
        elif bg_scale == "metacell_typed":                             # per-(host type, gene) sink + rho-binned POOL (tile-coupling fix)
            assert host_ctype is not None, "bg_scale='metacell_typed' needs host_ctype (M,) = host cell type index"
            NT = RHOt.shape[0]
            rho_max_g = RHOt.max(0).values                             # (G,) strongest expresser per gene
            rho_hm = RHOt[host_ctype, gp]                              # (M,) host's rho~ for the molecule's gene
            expr_m = rho_hm >= bg_alpha * rho_max_g[gp]                # host genuinely expresses g? (>= alpha*max)
            if bg_soft > 0:                                            # optional soft genuine-marker repel
                Kbg = Kbg * (1.0 - rho_hm / rho_max_g[gp].clamp_min(1e-30)).clamp(0.05, 1.0) ** bg_soft
            typ_key = torch.where(expr_m, host_ctype, torch.full_like(host_ctype, NT))   # NT = the shared POOL sink
            bg_flat = typ_key.to(torch.int64) * G + gp.to(torch.int64)  # sink key = (type | POOL, gene)
            bg_ub, bg_inv = torch.unique(bg_flat, return_inverse=True)
            uh, inv_h = torch.unique(host, return_inverse=True)        # unique cells -> area & type (each cell once)
            uarea = torch.zeros(uh.numel(), device=dev, dtype=area_host.dtype); uarea[inv_h] = area_host
            utype = torch.zeros(uh.numel(), dtype=torch.long, device=dev); utype[inv_h] = host_ctype
            A_t = torch.zeros(NT, device=dev, dtype=uarea.dtype).scatter_add_(0, utype, uarea)   # area per type in tile
            E = (RHOt >= bg_alpha * rho_max_g[None, :]).to(uarea.dtype)                          # (NT,G) expresser mask
            A_pool_g = (uarea.sum() - (E * A_t[:, None]).sum(0)).clamp_min(0.0)                  # (G,) non-expresser area/gene
            jt = bg_ub // G; gg = bg_ub % G
            A_key = torch.where(jt < NT, A_t[jt.clamp(max=NT - 1)], A_pool_g[gg])                # expresser->A_t ; POOL->A_pool(g)
            mu_bg_col = (kappa * lambda0[gg] * A_key).clamp_min(1e-12)
        else:                                                          # 'percell' (default): column = (host cell, gene)
            bg_flat = host.to(torch.int64) * G + gp                     # background column key = (host cell, gene)
            bg_ub, bg_inv = torch.unique(bg_flat, return_inverse=True)
            mu_bg_m = (kappa * lambda0[gp] * area_host).clamp_min(1e-12)     # mu_bg = kappa * lambda0[g] * area_host
            mu_bg_col = torch.zeros(bg_ub.numel(), device=dev); mu_bg_col[bg_inv] = mu_bg_m   # constant within a column
        v_bg = torch.ones(bg_ub.numel(), device=dev)

    for _ in range(niter):
        denom = (Km * vb[invK]).sum(1)
        if drop: denom = denom + Kbg * v_bg[bg_inv]
        u = 1.0 / denom.clamp_min(1e-30)
        colload = torch.zeros(ub.numel(), device=dev).scatter_add(0, inv, (u[:, None] * Km).reshape(-1))
        ratio = (target / colload.clamp_min(1e-30)).pow(pw)
        vb = torch.minimum(torch.ones_like(ratio), ratio) if budget == "nb" else ratio   # native: soft one-sided (nb)
        if drop:                                                        # background: FIRM one-sided onto mu_bg
            bg_load = torch.zeros(bg_ub.numel(), device=dev).scatter_add(0, bg_inv, u * Kbg)
            v_bg = torch.minimum(torch.ones_like(v_bg), (mu_bg_col / bg_load.clamp_min(1e-30)).pow(pw_bg))
    P = u[:, None] * Km * vb[invK]
    if not drop:
        return P.argmax(1).cpu().numpy()
    Pd = u * Kbg * v_bg[bg_inv]; win = P.argmax(1); nat_best = P.max(1).values
    hard = torch.where(Pd > nat_best, torch.full_like(win, -1), win).cpu().numpy()   # per-molecule arg-max decode
    if bg_scale == "neighborhood":                                   # Pass-1 EMIT: the fixed-radius window decode runs in run.py (engine.neighborhood_decode)
        assert host_slot is not None, "bg_scale='neighborhood' needs host_slot"
        desire = (u * Kbg).cpu().numpy(); natb = nat_best.cpu().numpy()
        margin = (desire - natb).astype(np.float32); win_np = win.cpu().numpy()
        eligible = (hard >= 0) & (win_np == host_slot.cpu().numpy()) & (margin > 0)   # cap-saved kept-at-host (window-drop candidates)
        soup_w = (desire / (desire + natb + 1e-30)).astype(np.float32)                # background responsibility vs best native, in (0,1) — soup-mass demand weight
        return hard, margin, eligible, soup_w
    if drop_decode != "leak":
        return hard
    # ---- 'leak' decode: recover the per-GENE leaked background mass the arg-max split away (matrix-faithful) ----
    desire = (u * Kbg).cpu().numpy()                                     # uncapped background desire = Pd / v_bg
    natb = nat_best.cpu().numpy(); pd_np = Pd.cpu().numpy(); gpn = gp.cpu().numpy().astype(np.int64)
    margin = desire - natb                                              # >0 => prefers bg absent the cap
    win_np = win.cpu().numpy()
    if host_slot is not None:                                           # KEPT = won by the host slot (exclude MOVES per the design)
        kept = (hard >= 0) & (win_np == host_slot.cpu().numpy())
    else:                                                              # fallback (run.py always passes host_slot): all non-dropped
        kept = (hard >= 0)
    ei = np.where(kept & (margin > 0))[0]                               # eligible = kept-but-would-drop-without-the-cap
    if len(ei):
        m_e = margin[ei]; pd_e = pd_np[ei]
        # leak budget is floored PER BUCKET: per (type|POOL, gene) SINK for metacell_typed (no cross-type spill), else per gene
        bucket = bg_flat.cpu().numpy()[ei] if bg_scale == "metacell_typed" else gpn[ei]
        uq, b = np.unique(bucket, return_inverse=True)                  # dense bucket index
        kg = np.floor(np.bincount(b, weights=pd_e) + 1e-9).astype(np.int64)   # budget = floor(sum Pd) per bucket (+eps guards FP under-floor)
        order = np.lexsort((-m_e, b))                                   # sort by bucket asc, then margin desc
        b_s = b[order]; rank = np.arange(len(b_s)) - np.searchsorted(b_s, b_s)          # within-bucket rank
        hard[ei[order][rank < kg[b_s]]] = -1                            # drop the top-floor(sum Pd) per bucket
    return hard


def neighborhood_decode(host_row, gene, margin, eligible, area, ctype, cxy, lambda0, RHO,
                        kappa=1.0, bg_alpha=0.12, radius=100.0, pix=10.0, min_expr_cells=2,
                        soup_w=None, demand="count", log=None, workers=-1):
    """Pass-2 window decode for bg_scale='neighborhood' — the tile-coupling de-blocking fix.

    Replaces the per-TILE leak floor (sinkhorn_solve drop_decode='leak') with a SLIDING fixed-RADIUS spatial window
    so the drop budget varies smoothly in space (no 500um grid blocks). Pass 1 (the per-tile Sinkhorn in run.py with
    bg_scale='neighborhood') emits, per CORE molecule, the per-cell physical-cap soft quantities; here we decide which
    ELIGIBLE molecules drop.

    Constraint per host-type|POOL x gene SINK s, on each cell's disk of radius `radius` (raw physical B_s capped ONLY
    by eligibility — the chosen operating point):
        A_s(c) = sum of same-sink cell AREA in the window      B_s(c) = kappa * lambda0[g] * A_s(c)   (physical capacity)
        E_s(c) = sum of same-sink eligible DEMAND in window    r_s(c) = min(1, B_s(c) / E_s(c))       (cap = eligibility)
                 demand='count': each eligible molecule counts 1 (conservative). demand='mass': each counts its soup
                 responsibility w=desire/(desire+best_native) in (0,1) -> smaller E_s -> larger r_s -> cleans more (~ typed's soup mass).
    A molecule drops iff its GLOBAL per-sink margin rank-fraction < r_s(c): each cell sheds the top-r_s(c) fraction of
    its sink by background-margin, where r_s(c) is the LOCAL window budget rate. No per-cell floor(r*n) -> recovers the
    sub-unit tail (a per-cell floor zeroes sparse genes like CEACAM6 with ~1 molecule/cell -> under-cleans). r_s(c) is
    a sliding-window field so the drop fraction is smooth in space. The sink key (expresser type if
    rho~[t,g] >= bg_alpha*max_t rho~ else POOL) is the SAME global key as metacell_typed, so the cross-type-spill fix
    is retained. Window sums are computed on a `pix`-um grid by disk convolution. Genes with < min_expr_cells eligible
    molecules are left untouched (r=0, the safe under-clean direction).

    host_row/gene/margin/eligible are per-CORE-molecule (N,); area/ctype/cxy/lambda0/RHO are per-cell tables.
    Returns drop (N,) bool — True only where an eligible molecule should be DROPPED.

    Window sums use a cached-kernel BATCHED real FFT that is BIT-FOR-BIT identical to the per-call
    scipy.signal.fftconvolve(grid, ker, 'same') it replaces (the disk spectrum is transformed once, the
    cell->pixel scatter is np.bincount over only the nonzero cells = same accumulation order as np.add.at, and a
    whole gene-chunk's window grids share one multithreaded rfftn, workers=-1 — also bit-identical to workers=1).
    Result-identical, far faster on large panels; regression-tested in tests/test_neighborhood_decode.py.
    """
    import scipy.fft as sfft
    _log = log or (lambda *a: None)
    N = len(host_row); NCELL = len(area); NT, G = RHO.shape
    drop = np.zeros(N, dtype=bool)
    ei = np.where(eligible)[0]
    if len(ei) == 0: return drop
    ec = host_row[ei].astype(np.int64); eg = gene[ei].astype(np.int64); em = margin[ei].astype(np.float64)
    ew = soup_w[ei].astype(np.float64) if (demand == "mass" and soup_w is not None) else None   # per-eligible soup weight (mass demand)

    # ---- grid + disk kernel for fixed-radius window sums ----
    x0 = float(cxy[:, 0].min()); y0 = float(cxy[:, 1].min())
    gx = np.clip(((cxy[:, 0] - x0) / pix).astype(np.int64), 0, None)
    gy = np.clip(((cxy[:, 1] - y0) / pix).astype(np.int64), 0, None)
    nx = int(gx.max()) + 1; ny = int(gy.max()) + 1
    rpix = max(int(round(radius / pix)), 1)
    yy, xx = np.ogrid[-rpix:rpix + 1, -rpix:rpix + 1]; ker = (xx * xx + yy * yy <= rpix * rpix).astype(np.float64)
    lin = (gx * ny + gy).astype(np.int64); npix = nx * ny             # per-cell flat pixel index (row-major == (gx,gy))
    # Cached-kernel disk convolution: replicate scipy.signal.fftconvolve(grid, ker, 'same') EXACTLY (full-conv shape,
    # next_fast_len, centered crop) but transform the kernel ONCE and batch many signal grids in a single rfftn.
    sh0, sh1 = nx + ker.shape[0] - 1, ny + ker.shape[1] - 1
    fshape = (int(sfft.next_fast_len(sh0, True)), int(sfft.next_fast_len(sh1, True)))
    K_fft = sfft.rfftn(ker, fshape, workers=workers)                  # disk spectrum, reused for every window sum
    def _scatter(idx, w):                                             # sparse cell->pixel scatter (omitting zero cells is exact)
        return np.bincount(lin[idx], weights=w, minlength=npix).reshape(nx, ny)
    def winsum(vals):                                                 # dense per-cell vector -> windowed grid (area helpers; few calls)
        g_ = np.bincount(lin, weights=vals.astype(np.float64), minlength=npix).reshape(nx, ny)
        return sfft.irfftn(sfft.rfftn(g_, fshape, workers=workers) * K_fft, fshape, workers=workers)[rpix:rpix + nx, rpix:rpix + ny]
    def winconv_batch(grids):                                         # (M,nx,ny) scattered grids -> (M,nx,ny) windowed, one batched rfftn
        SP = sfft.rfftn(grids, fshape, axes=(1, 2), workers=workers)  # workers=-1 (all cores) is bit-identical to workers=1 (tested)
        return sfft.irfftn(SP * K_fft[None], fshape, axes=(1, 2), workers=workers)[:, rpix:rpix + nx, rpix:rpix + ny]
    Awin_total = winsum(area)                                         # window area of ALL cells (POOL helper); gene-independent
    _area_win_t = {}                                                  # type -> window-area grid (cached across genes)
    def area_win_type(t):
        if t not in _area_win_t: _area_win_t[t] = winsum(area * (ctype == t))
        return _area_win_t[t]

    rho_max_g = RHO.max(0); E = RHO >= bg_alpha * rho_max_g[None, :]   # (NT,G) expresser mask (global, == metacell_typed key)
    r_per = np.zeros(len(ei), dtype=np.float64)
    og = np.argsort(eg, kind="stable"); ugen, gstart = np.unique(eg[og], return_index=True)
    gstart = np.append(gstart, len(eg))
    # Genes are processed in chunks; ALL per-(gene[, expresser-type]) eligible window grids in a chunk are transformed
    # by ONE batched rfftn. job_budget bounds the batch memory (~ jobs * npix * 8 bytes for the stacked grids).
    job_budget = int(np.clip(4e7 // max(npix, 1), 8, 512))
    gk = 0
    while gk < len(ugen):
        chunk = []; jobs = []                                        # jobs: scattered (nx,ny) grids awaiting convolution
        while gk < len(ugen) and len(jobs) < job_budget:
            g = int(ugen[gk]); sl = og[gstart[gk]:gstart[gk + 1]]; cells_g = ec[sl]   # positions (ei-space) of this gene's molecules
            if len(sl) < min_expr_cells: gk += 1; continue           # too few -> r=0 (no drop; safe)
            uc, inv = np.unique(cells_g, return_inverse=True); inv = inv.reshape(-1)  # compact cells; inv == searchsorted(uc, cells_g)
            ev = np.bincount(inv, weights=(ew[sl] if ew is not None else None), minlength=len(uc)).astype(np.float64)  # per-cell eligible demand (mass: Σ soup w; count: #)
            ct_uc = ctype[uc]
            expr_types = np.where(E[:, g])[0]
            tot_job = len(jobs); jobs.append(_scatter(uc, ev))       # total eligible window (== winsum(elig_cnt); POOL helper)
            type_job = {}
            for t in expr_types:                                     # per-expresser-type eligible window (== winsum(elig_cnt*(ctype==t)))
                mt = ct_uc == int(t); type_job[int(t)] = len(jobs); jobs.append(_scatter(uc[mt], ev[mt]))
            chunk.append((g, sl, inv, uc, expr_types, tot_job, type_job))
            gk += 1
        if not jobs: continue
        conv = winconv_batch(np.stack(jobs))                         # ONE batched rfftn for the whole chunk
        for g, sl, inv, uc, expr_types, tot_job, type_job in chunk:
            gxx = gx[uc]; gyy = gy[uc]; own = ctype[uc]; is_expr = E[own, g]
            A_same = Awin_total[gxx, gyy].copy(); E_same = conv[tot_job][gxx, gyy].copy()   # default = POOL (totals - expressers)
            sumA = np.zeros(len(uc)); sumE = np.zeros(len(uc)); ew_t = {}
            for t in expr_types:
                aw = area_win_type(int(t))[gxx, gyy]; ewg = conv[type_job[int(t)]][gxx, gyy]
                ew_t[int(t)] = (aw, ewg); sumA += aw; sumE += ewg
            A_same -= sumA; E_same -= sumE                           # POOL same-sink area/eligibles = total - sum_expresser
            for t in expr_types:                                     # expresser cells use their OWN type's window
                m = is_expr & (own == int(t)); aw, ewg = ew_t[int(t)]
                A_same[m] = aw[m]; E_same[m] = ewg[m]
            r_uc = np.minimum(1.0, kappa * float(lambda0[g]) * np.maximum(A_same, 0.0) / np.maximum(E_same, 1e-9))
            r_per[sl] = r_uc[inv]                                     # inv == searchsorted(uc, cells_g) (uc sorted unique)
    # ---- drop: GLOBAL per-sink margin rank gated by the LOCAL windowed rate r_s(c). No per-cell floor -> recovers the
    #      sub-unit tail (a per-cell floor zeroes sparse genes -> under-cleans); smooth because r_s(c) is a window field. ----
    own = ctype[ec]; tkey = np.where(E[own, eg], own, NT).astype(np.int64)   # sink = (expresser type | POOL, gene), == metacell_typed key
    sink = tkey * G + eg
    order = np.lexsort((-em, sink)); ss = sink[order]
    rank = np.arange(len(ss)) - np.searchsorted(ss, ss)             # 0-based GLOBAL margin rank within the sink (highest margin = rank 0)
    _, inv_s, cnt_s = np.unique(ss, return_inverse=True, return_counts=True); n_sink = cnt_s[inv_s]
    rank_frac = rank / np.maximum(n_sink, 1)                        # margin percentile-from-top within the sink, in [0,1)
    drop[ei[order[rank_frac < r_per[order]]]] = True               # shed the top-r_s(c) fraction by margin (r_s = local window budget)
    _log(f"  neighborhood_decode: dropped {int(drop.sum()):,}/{len(ei):,} eligible "
         f"({100 * drop.sum() / max(len(ei), 1):.1f}%) over {len(ugen)} genes, radius={radius}um pix={pix}um")
    return drop
