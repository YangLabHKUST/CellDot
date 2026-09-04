"""Stage 2 — tile-Sinkhorn solver. Reads the prep artefacts and writes the cleaned data + provenance.

Parameterised by ``CellDotConfig``. Reads the gamma-corrected prior (raw fallback), cells_index, transcript
shards, ambient soup, and meta from ``cfg.out``; writes:
  cleaned.h5ad       X=raw counts; layers raw / greedy / celldot; obs indexed by the ORIGINAL cell_id (type,
                     centroid, per-cell fate counts); var genes (lambda0); uns['celldot'] provenance
  molecules.parquet  per CORE molecule: x, y, gene (name), old_host (cell_id), new_host (cell_id, or "" when the
                     molecule was removed as background), action in keep / move / drop
Cells are identified ONLY by their original cell_id; integer positions exist in memory only.

Geometry is centroid-minus-radius (d = dist_to_centroid - cell_radius, clipped at 0). The true-membrane
distance path exists in the engine (``tile_distance(mode='boundary')``) but is intentionally NOT wired in
yet — see the design doc; this keeps the solver identical to the validated benchmark run.

``greedy`` = per-molecule argmax affinity (no budget, no drop) — the reference ceiling baseline.
``celldot`` = NB over-dispersed native budget (Z) + per-cell PHYSICAL background mu_bg = kappa*lambda0*area
(firm one-sided column), sparse Sinkhorn.
"""
import os, json, time, glob, hashlib, numpy as np, pandas as pd
import pyarrow as pa, pyarrow.parquet as pq
from scipy.spatial import cKDTree
from scipy import sparse
import torch, anndata as ad
from . import engine


def run(cfg):
    t0 = time.time(); log = lambda *a: print(f"[run {time.time()-t0:6.1f}s]", *a, flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    meta = json.load(open(cfg.meta)); N_SOUP = meta.get("n_soup", 0)
    K, R, ELL, Z, PW = cfg.K, cfg.R, cfg.ELL, cfg.Z, cfg.PW; KAPPA, PW_BG = cfg.KAPPA, cfg.PW_BG
    DROP_DECODE = getattr(cfg, "DROP_DECODE", "leak"); BG_SCALE = getattr(cfg, "BG_SCALE", "metacell")
    BG_ALPHA = getattr(cfg, "BG_ALPHA", 0.12); BG_SOFT = getattr(cfg, "BG_SOFT", 0.0)   # metacell_typed knobs
    NB = (BG_SCALE == "neighborhood")                                                    # sliding-radius window decode (de-block)
    BG_RADIUS = getattr(cfg, "BG_RADIUS", 100.0); BG_PIX = getattr(cfg, "BG_PIX", 10.0)  # neighborhood knobs
    BG_DEMAND = getattr(cfg, "BG_DEMAND", "count")                                        # neighborhood drop-rate denominator: 'count' | 'mass'
    TILE, HALO, NITER, EPS = cfg.TILE, cfg.HALO, cfg.NITER, cfg.EPS

    rho_f = cfg.rho_corrected if os.path.exists(cfg.rho_corrected) else cfg.rho   # gamma-corrected prior; raw fallback
    rho = pd.read_parquet(rho_f); log(f"CellDot dev={dev} prior={os.path.basename(rho_f)} ELL={ELL} pw={PW} Z={Z} kappa={KAPPA} pw_bg={PW_BG}")
    genes = list(rho.columns); types = list(rho.index); tpos = {t: i for i, t in enumerate(types)}
    RHO = np.clip(rho.values.astype(np.float32), 1e-6, None); G = len(genes); NT = len(types)
    ci = pd.read_parquet(cfg.cells_index); cell_ids = ci["cell_id"].astype(str).values          # file order = internal position
    cid2pos = pd.Index(cell_ids); assert cid2pos.is_unique and "" not in cid2pos, "cell_id must be unique and non-empty"
    cxy = ci[["x_centroid", "y_centroid"]].values.astype(float); typ = ci["type"].astype(str).values
    ctype = np.array([tpos.get(t, -1) for t in typ], int); NCELL = len(ci)
    assert (ctype >= 0).all(), f"{int((ctype < 0).sum())} cells have a type absent from rho — annotate/mask them (else RHO[-1] mis-indexes)"
    _ca = pd.read_parquet(cfg.CELLS, columns=["cell_id", "cell_area"])
    carea = pd.Series(_ca["cell_area"].values, index=_ca["cell_id"].astype(str).values)
    area = np.maximum(carea.reindex(cell_ids).fillna(1.0).values.astype(float), 1.0); cr = np.sqrt(area / np.pi)
    tree = cKDTree(cxy); RHOt = torch.tensor(RHO, device=dev)
    area_t = torch.tensor(area, device=dev, dtype=torch.float32)
    # per-cell PHYSICAL background — DIRECT extracellular counts (clean design): n_extra_g = the raw per-gene
    # UNASSIGNED count from prep (NO a_g·N_soup reconstruction, NO panel-mean shrink). Two roles from one count:
    #   p_bg[g] = n_extra_g / Σn_extra  (+tiny pseudocount → keeps p_bg>0, so a no-in-range-candidate molecule drops)
    #   λ⁰[g]   = n_extra_g / A_extra   (A_extra = tissue-mask area − Σcell_area, from prep; grid fallback for legacy)
    amb = pd.read_parquet(cfg.ambient).set_index("gene")
    if "n_extra" in amb.columns:
        n_extra = amb["n_extra"].reindex(genes).fillna(0.0).values.astype(np.float64)
    else:                                                             # legacy parquet → reconstruct from shrunk a_g
        n_extra = amb["a_g"].reindex(genes).fillna(0.0).values.astype(np.float64) * float(N_SOUP)
    A_extra = float(meta.get("A_extra", 0.0))
    if A_extra <= 1.0:                                                # legacy meta → 30µm centroid-grid occupancy fallback
        _bx = np.floor((cxy[:, 0] - cxy[:, 0].min()) / 30.0).astype(np.int64); _by = np.floor((cxy[:, 1] - cxy[:, 1].min()) / 30.0).astype(np.int64)
        A_extra = max(len(np.unique(_bx * (_by.max() + 1) + _by)) * 900.0 - float(area.sum()), 1.0)
    lambda0, p_bg = engine.background_params(n_extra, A_extra)        # clean design §4 (shared with the experiment driver)
    LAM0 = torch.tensor(lambda0, device=dev); PBG = torch.tensor(p_bg, device=dev)

    XS, YS, GS, HS = [], [], [], []
    for sh in sorted(glob.glob(cfg.assign_dir + "/shard_*.parquet")):
        d = pd.read_parquet(sh, columns=["x", "y", "gene", "cell_id"])
        XS.append(d.x.values.astype(np.float32)); YS.append(d.y.values.astype(np.float32))
        GS.append(d.gene.values.astype(np.int64))
        h = d["cell_id"]; h = h if isinstance(h.dtype, pd.CategoricalDtype) else h.astype(str).astype("category")
        pos = cid2pos.get_indexer(h.cat.categories.astype(str)); assert (pos >= 0).all(), f"{sh}: cell_id not in cells_index"
        HS.append(pos[h.cat.codes.values.astype(np.int64)].astype(np.int64))   # cell_id -> in-memory position
    xa, ya, ga, ha = map(np.concatenate, (XS, YS, GS, HS))
    Ntx = len(xa); raw = sparse.coo_matrix((np.ones(Ntx, np.float32), (ha, ga)), shape=(NCELL, G)).tocsr()
    Nraw_all = np.asarray(raw.sum(1)).ravel().astype(np.float32)
    Ften = torch.tensor(engine.estimate_fano(raw, ctype, NT, G, clip=(cfg.fano_lo, cfg.fano_hi)), device=dev)
    log(f"cells {NCELL} tx {Ntx:,} genes {G} types {NT} | A_extra {A_extra:.3e} ({'mask' if meta.get('A_extra',0)>1 else 'grid'}) "
        f"ambient {n_extra.sum()/max(A_extra,1):.4f}/um^2 mean mu_bg/cell@k={KAPPA} {lambda0.sum()*area.mean()*KAPPA:.1f} "
        f"soup_top {genes[int(n_extra.argmax())]}")

    def build_tile(box):
        xlo, xhi, ylo, yhi = box
        sel = (xa >= xlo - HALO) & (xa < xhi + HALO) & (ya >= ylo - HALO) & (ya < yhi + HALO)
        if not sel.any(): return None
        xp, yp, gp, hp = xa[sel], ya[sel], ga[sel], ha[sel]
        core = (xp >= xlo) & (xp < xhi) & (yp >= ylo) & (yp < yhi)
        if not core.any(): return None
        M = len(xp); pts = np.column_stack([xp, yp]); dist, idx = tree.query(pts, k=K)
        d = np.clip(dist - cr[idx], 0, None).astype(np.float32); within = (dist <= R).astype(np.float32)
        aff = (np.log(RHO[ctype[idx], gp[:, None]]) - d / ELL).astype(np.float32); aff[within == 0] = -30.0
        U, inv = np.unique(idx.ravel(), return_inverse=True); candU = inv.reshape(idx.shape)
        hpt = torch.tensor(hp, device=dev, dtype=torch.long)
        hmatch = idx == hp[:, None]                                     # which candidate slot is the host cell
        host_slot = np.where(hmatch.any(1), hmatch.argmax(1), -1).astype(np.int64)   # -1 if host not among K candidates
        return dict(M=M, xp=xp, yp=yp, gp=gp, hp=hp, idx=idx, core=core, aff_np=aff,
                    aff=torch.tensor(aff, device=dev), within=torch.tensor(within, device=dev),
                    candU=torch.tensor(candU, device=dev), gpt=torch.tensor(gp, device=dev),
                    ctypeU=torch.tensor(ctype[U], device=dev, dtype=torch.long),
                    NrawU=torch.tensor(Nraw_all[U], device=dev, dtype=torch.float32),
                    host=hpt, area_host=area_t[hpt], host_slot=torch.tensor(host_slot, device=dev, dtype=torch.long),
                    host_ctype=torch.tensor(ctype[hp], device=dev, dtype=torch.long))   # host cell type (metacell_typed)

    x0g, y0g = xa.min(), ya.min(); nx = int((xa.max() - x0g) / TILE) + 1; ny = int((ya.max() - y0g) / TILE) + 1
    acc = {"greedy": ([], []), "celldot": ([], [])}; ndrop = ncore = ntile = 0
    rx, ry, rgn, rold, rnew = [], [], [], [], []                       # molecule reassignment record (CORE molecules)
    nbH = nbG = nbD = nbM = nbE = nbX = nbY = nbW = None
    if NB: nbH, nbG, nbD, nbM, nbE, nbX, nbY, nbW = [], [], [], [], [], [], [], []   # neighborhood Pass-1 buffers: host,gene,dest,margin,eligible,x,y,soup_w
    for ti in range(nx):
        for tj in range(ny):
            box = (x0g + ti * TILE, x0g + (ti + 1) * TILE, y0g + tj * TILE, y0g + (tj + 1) * TILE)
            T = build_tile(box)
            if T is None: continue
            ar = np.arange(T["M"]); core = T["core"]; idx = T["idx"]; gp = T["gp"]
            gw = T["aff_np"].argmax(1)
            acc["greedy"][0].append(idx[ar[core], gw[core]]); acc["greedy"][1].append(gp[core])
            out = engine.sinkhorn_solve(T["aff"], T["within"], T["candU"], T["gpt"], T["ctypeU"], T["NrawU"], RHOt, Ften, G, dev,
                                      lambda0=LAM0, host=T["host"], area_host=T["area_host"],
                                      budget="nb", pw=PW, drop=True, Z=Z, eps=EPS, niter=NITER, kappa=KAPPA, pw_bg=PW_BG, p_bg=PBG,
                                      drop_decode=DROP_DECODE, host_slot=T["host_slot"], bg_scale=BG_SCALE,
                                      host_ctype=T["host_ctype"], bg_alpha=BG_ALPHA, bg_soft=BG_SOFT)
            co = np.where(core)[0]                                     # provenance for every core molecule
            if NB:                                                    # neighborhood: buffer Pass-1 soft data; the drop is decided after the loop
                w, marg, elig, sw = out; hard_co = w[co]
                dest = np.where(hard_co >= 0, idx[co, np.clip(hard_co, 0, None)], -1).astype(np.int64)   # argmax destination (host/neighbour/-1)
                nbH.append(T["hp"][co].astype(np.int64)); nbG.append(gp[co].astype(np.int64)); nbD.append(dest)
                nbM.append(marg[co]); nbE.append(elig[co]); nbX.append(T["xp"][co]); nbY.append(T["yp"][co]); nbW.append(sw[co])
            else:
                w = out
                kp = core & (w >= 0); acc["celldot"][0].append(idx[ar[kp], w[kp]]); acc["celldot"][1].append(gp[kp])
                newh = np.where(w[co] >= 0, idx[co, np.clip(w[co], 0, None)], -1).astype(np.int64)
                rx.append(T["xp"][co]); ry.append(T["yp"][co]); rgn.append(gp[co]); rold.append(T["hp"][co]); rnew.append(newh)
                ndrop += int((core & (w < 0)).sum())
            ncore += int(core.sum()); ntile += 1
            if ntile % 20 == 0: log(f"  tile {ntile} ({time.time()-t0:.0f}s)")
    if NB:                                                            # ---- Pass 2: fixed-radius WINDOW decode (replaces the per-tile leak floor -> no 500um grid blocks) ----
        host_all = np.concatenate(nbH); gene_all = np.concatenate(nbG); dest_all = np.concatenate(nbD)
        marg_all = np.concatenate(nbM); elig_all = np.concatenate(nbE)
        x_all = np.concatenate(nbX); y_all = np.concatenate(nbY); sw_all = np.concatenate(nbW)
        log(f"neighborhood Pass-2: {len(host_all):,} core-tx, {int(elig_all.sum()):,} eligible | radius={BG_RADIUS}um pix={BG_PIX}um demand={BG_DEMAND}")
        drop_e = engine.neighborhood_decode(host_all, gene_all, marg_all, elig_all,
                    area.astype(np.float32), ctype.astype(np.int64), cxy.astype(np.float32),
                    lambda0.astype(np.float32), RHO.astype(np.float32),
                    kappa=KAPPA, bg_alpha=BG_ALPHA, radius=BG_RADIUS, pix=BG_PIX,
                    soup_w=sw_all, demand=BG_DEMAND, log=log)
        newh_all = np.where(drop_e, -1, dest_all).astype(np.int64)   # eligible+window-drop -> -1; else keep the Pass-1 destination
        kpm = newh_all >= 0
        acc["celldot"] = ([newh_all[kpm]], [gene_all[kpm]])        # cell x gene matrix rows/cols (kept molecules)
        rx, ry, rgn, rold, rnew = [x_all], [y_all], [gene_all], [host_all], [newh_all]
        ndrop = int((newh_all < 0).sum())
    log(f"tiled {ntile} core-tx {ncore:,} drop-frac {ndrop/max(ncore,1):.3f}")

    def mat(rs, cs):
        return sparse.coo_matrix((np.ones(sum(map(len, rs)), np.float32), (np.concatenate(rs), np.concatenate(cs))),
                                 shape=(NCELL, G)).tocsr()
    # molecule fate table — every cell named by its ORIGINAL cell_id (dictionary-encoded strings; "" = no cell,
    # i.e. removed as background); gene by name. Positions are used only to build it.
    _oh = np.concatenate(rold).astype(np.int64); _nh = np.concatenate(rnew).astype(np.int64)
    _gn = np.concatenate(rgn).astype(np.int64)
    host_cats = np.concatenate([[""], cell_ids])                               # code 0 = "" (dropped)
    mol = pd.DataFrame({"x": np.concatenate(rx), "y": np.concatenate(ry),
                        "gene": pd.Categorical.from_codes(_gn, categories=genes),
                        "old_host": pd.Categorical.from_codes(_oh + 1, categories=host_cats),
                        "new_host": pd.Categorical.from_codes(_nh + 1, categories=host_cats),
                        "action": pd.Categorical(np.where(_nh < 0, "drop", np.where(_nh == _oh, "keep", "move")),
                                                 categories=["keep", "move", "drop"])})

    # per-cell PHYSICAL background budget + realized shed — saved into obs so cleaned.h5ad is self-contained
    # (no need to re-derive from molecules.parquet). mu_bg = the expected ambient tx in the cell footprint.
    mu_bg_cell = (KAPPA * float(lambda0.sum()) * area).astype(np.float32)     # = kappa * area * sum_g lambda0  [tx]
    _md = _nh < 0; _mv = (~_md) & (_nh != _oh)
    n_host_c = np.bincount(_oh, minlength=NCELL); n_drop_c = np.bincount(_oh[_md], minlength=NCELL)
    n_moveout_c = np.bincount(_oh[_mv], minlength=NCELL); n_movein_c = np.bincount(_nh[_mv], minlength=NCELL)
    obs = pd.DataFrame({"type": pd.Categorical(typ),
                        "x_centroid": cxy[:, 0], "y_centroid": cxy[:, 1], "mu_bg": mu_bg_cell,
                        "n_dropped": n_drop_c.astype(np.int32), "n_moved_out": n_moveout_c.astype(np.int32),
                        "n_moved_in": n_movein_c.astype(np.int32),
                        "drop_frac": (n_drop_c / np.maximum(n_host_c, 1)).astype(np.float32)},
                       index=pd.Index(cell_ids, name="cell_id"))                 # obs_names = the original cell_id
    var = pd.DataFrame({"lambda0": lambda0}, index=genes)                     # per-gene free-ambient areal density [tx/um^2]
    A = ad.AnnData(X=raw, obs=obs, var=var)
    A.layers["raw"] = raw; A.layers["greedy"] = mat(*acc["greedy"]); A.layers["celldot"] = mat(*acc["celldot"])

    # ---- provenance: ONE record written into BOTH cleaned.h5ad (uns['celldot']) AND molecules.parquet (schema
    #      metadata key b'celldot'), so the cleaned layer and its per-molecule fate can NEVER silently drift.
    #      run_id = md5(full params + result stats); identical inputs -> identical run_id, and the h5ad/parquet
    #      pair from one run() always share it. Verify a pair with read_provenance(...)['run_id']. ----
    try:
        from . import __version__ as PKG_VERSION
    except Exception:
        PKG_VERSION = "unknown"
    vc = mol.action.value_counts().to_dict()
    params = dict(K=K, R=R, ELL=ELL, Z=Z, pw=PW, kappa=KAPPA, pw_bg=PW_BG, drop_decode=DROP_DECODE,
                  bg_scale=BG_SCALE, bg_alpha=BG_ALPHA, bg_soft=BG_SOFT, bg_radius=BG_RADIUS, bg_pix=BG_PIX,
                  bg_demand=BG_DEMAND, bg_dilate=float(getattr(cfg, "BG_DILATE", 0.0)),
                  tile=TILE, halo=HALO, niter=NITER, eps=EPS, qv=float(getattr(cfg, "qv", 0.0)),
                  lam=float(getattr(cfg, "lam", 0.0)), min_prob=float(getattr(cfg, "min_prob", 0.0)),
                  A_extra=round(A_extra, 1))
    spd = dict(dataset=cfg.dataset, version=PKG_VERSION, n_tx=int(Ntx),
               drop_frac=round(ndrop / max(ncore, 1), 4), ambient_density=round(float(N_SOUP / max(A_extra, 1)), 5),
               n_cells=int(NCELL), n_genes=int(G), n_types=int(NT),
               n_keep=int(vc.get("keep", 0)), n_move=int(vc.get("move", 0)), n_drop=int(vc.get("drop", 0)),
               params=params)
    spd["run_id"] = hashlib.md5(json.dumps(spd, sort_keys=True, default=str).encode()).hexdigest()[:16]
    A.uns["celldot"] = spd
    A.write(cfg.cleaned)

    tbl = pa.Table.from_pandas(mol, preserve_index=False)                     # molecules.parquet WITH matching provenance
    md = dict(tbl.schema.metadata or {}); md[b"celldot"] = json.dumps(spd, default=str).encode()
    pq.write_table(tbl.replace_schema_metadata(md), cfg.molecules)

    log(f"wrote {cfg.cleaned} + {cfg.molecules}  run_id={spd['run_id']} v{PKG_VERSION}  (mol {len(mol):,}: "
        f"keep {vc.get('keep',0):,} move {vc.get('move',0):,} drop {vc.get('drop',0):,})")
    log("DONE run")
    return A


def read_provenance(path):
    """Return the provenance dict recorded in a run's output — accepts either a cleaned .h5ad
    (``uns['celldot']``) or a molecules .parquet (schema metadata key ``b'celldot'``). Use it to confirm a cleaned
    layer and its molecule-fate file are from the SAME run::

        read_provenance(cfg.cleaned)["run_id"] == read_provenance(cfg.molecules)["run_id"]
    """
    if str(path).endswith(".parquet"):
        raw = (pq.read_schema(path).metadata or {}).get(b"celldot")
        return json.loads(raw) if raw else {}
    A = ad.read_h5ad(path, backed="r")
    u = A.uns.get("celldot", {})
    try:
        return dict(u)
    except Exception:
        return u
