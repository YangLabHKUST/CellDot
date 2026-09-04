"""Unit test for engine.neighborhood_decode vs an independent brute-force reference, for BOTH demand modes
('count' and 'mass'). Cells on DISTINCT integer pixels so the grid/disk-convolution window == the brute-force
pixel-disk window EXACTLY -> any mismatch is a logic bug. Also checks mass-demand drops >= count-demand.

Run:  python tests/test_neighborhood_decode.py    (needs numpy + scipy; torch is stubbed, not required)
"""
import os, sys, types, importlib.util
sys.modules.setdefault("torch", types.ModuleType("torch"))     # engine imports torch but neighborhood_decode never calls it
import numpy as np
_eng_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "celldot", "engine.py")
_spec = importlib.util.spec_from_file_location("spd_engine", _eng_path)
_eng = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_eng)
neighborhood_decode = _eng.neighborhood_decode

PIX = 10.0; RADIUS = 50.0; KAPPA = 1.0; ALPHA = 0.12


def ref_decode(host_row, gene, margin, eligible, area, ctype, cxy, lambda0, RHO, kappa, alpha, radius, pix,
               min_expr_cells=2, soup_w=None, demand="count"):
    N = len(host_row); NCELL = len(area); NT, G = RHO.shape
    drop = np.zeros(N, bool); ei = np.where(eligible)[0]
    if len(ei) == 0: return drop
    ec = host_row[ei]; eg = gene[ei]; em = margin[ei]
    ew = soup_w[ei] if (demand == "mass" and soup_w is not None) else None
    gx = np.round((cxy[:, 0] - cxy[:, 0].min()) / pix).astype(int); gy = np.round((cxy[:, 1] - cxy[:, 1].min()) / pix).astype(int)
    rpx = max(int(round(radius / pix)), 1)
    dxx = gx[:, None] - gx[None, :]; dyy = gy[:, None] - gy[None, :]; Adj = (dxx * dxx + dyy * dyy) <= rpx * rpx
    rho_max = RHO.max(0); E = RHO >= alpha * rho_max[None, :]; r_per = np.zeros(len(ei))
    for g in np.unique(eg):
        sl = np.where(eg == g)[0]
        if len(sl) < min_expr_cells: continue
        wcell = np.zeros(NCELL); np.add.at(wcell, ec[sl], (ew[sl] if ew is not None else 1.0))
        for k in sl:
            c = ec[k]; win = Adj[c]
            same_expr = E[ctype, g]
            mask = win & (ctype == ctype[c]) if E[ctype[c], g] else win & (~same_expr)
            A = area[mask].sum(); Ecnt = wcell[mask].sum()
            r_per[k] = min(1.0, kappa * lambda0[g] * max(A, 0.0) / max(Ecnt, 1e-9))
    own = ctype[ec]; tkey = np.where(E[own, eg], own, NT).astype(np.int64); sink = tkey * G + eg
    for s in np.unique(sink):
        ix = np.where(sink == s)[0]; o = ix[np.argsort(-em[ix], kind="stable")]
        rank_frac = np.arange(len(o)) / max(len(o), 1); drop[ei[o[rank_frac < r_per[o]]]] = True
    return drop


def make_case(seed, lam_scale, collide=False):
    rng = np.random.default_rng(seed)
    if collide:                                          # MANY cells per pixel (exercises the bincount scatter-order path)
        NCELL = 200; NT, G = 3, 4; side = 9              # 81 pixels, 200 cells -> ~2-3 cells share a pixel
        pixels = rng.integers(0, side * side, NCELL); cgx, cgy = pixels // side, pixels % side
        RHO = np.array([[1.0, 1.0, 0.001, 1.0],          # gene expressers: {0} / {0,1} / {1,2} / {0,1,2}  (POOL + multi-expresser)
                        [0.001, 1.0, 1.0, 1.0],
                        [0.001, 0.001, 1.0, 1.0]], np.float32)
        lambda0 = np.full(G, 0.01 * lam_scale, np.float32)
    else:                                                # cells on DISTINCT pixels (window == brute-force pixel disk exactly)
        NCELL = 200; NT, G = 2, 2
        pixels = rng.choice(30 * 30, NCELL, replace=False); cgx, cgy = pixels // 30, pixels % 30
        RHO = np.array([[1.0, 0.001], [0.001, 1.0]], np.float32); lambda0 = np.array([0.01 * lam_scale, 0.01 * lam_scale], np.float32)
    cxy = np.stack([cgx * PIX, cgy * PIX], 1).astype(np.float32)   # exact PIX multiples -> engine trunc == ref round
    area = rng.uniform(50, 150, NCELL).astype(np.float32); ctype = rng.integers(0, NT, NCELL)
    M = 1500
    host = rng.integers(0, NCELL, M).astype(np.int64); gene = rng.integers(0, G, M).astype(np.int64)
    margin = rng.uniform(0.1, 5.0, M).astype(np.float32); eligible = rng.random(M) < 0.8
    soup_w = rng.uniform(0.5, 1.0, M).astype(np.float32)
    return dict(host_row=host, gene=gene, margin=margin, eligible=eligible, area=area, ctype=ctype, cxy=cxy,
                lambda0=lambda0, RHO=RHO, soup_w=soup_w)


def _nd(c, dem, workers=-1):
    return neighborhood_decode(c["host_row"], c["gene"], c["margin"], c["eligible"], c["area"], c["ctype"],
                               c["cxy"], c["lambda0"], c["RHO"], kappa=KAPPA, bg_alpha=ALPHA, radius=RADIUS,
                               pix=PIX, soup_w=c["soup_w"], demand=dem, workers=workers)


def main():
    ok = True
    for collide in [False, True]:                              # sparse (distinct-pixel) AND colliding (many cells/pixel)
        for seed in range(6):
            for lam_scale, tag in [(1.0, "mixed-r"), (100.0, "saturate"), (0.0, "r=0")]:
                c = make_case(seed, lam_scale, collide=collide); dc = {}
                for dem in ["count", "mass"]:
                    got = _nd(c, dem)
                    exp = ref_decode(c["host_row"], c["gene"], c["margin"], c["eligible"], c["area"], c["ctype"], c["cxy"],
                                     c["lambda0"], c["RHO"], KAPPA, ALPHA, RADIUS, PIX, soup_w=c["soup_w"], demand=dem)
                    m = np.array_equal(got, exp); ok &= m; dc[dem] = int(got.sum())
                massge = dc["mass"] >= dc["count"]; ok &= massge
                print(f"{'collide' if collide else 'sparse ':7s} seed{seed} {tag:9s}: count {dc['count']:4d} | mass {dc['mass']:4d} "
                      f"| exact-match {m} | mass>=count {'PASS' if massge else 'FAIL'}")
    # multithreaded (workers=-1) must be BIT-IDENTICAL to single-thread (workers=1) — the batched-rFFT speedup is result-safe
    we = True
    for collide in [False, True]:
        c = make_case(3, 1.0, collide=collide)
        for dem in ["count", "mass"]: we &= np.array_equal(_nd(c, dem, workers=1), _nd(c, dem, workers=-1))
    ok &= we; print(f"\nworkers=1 == workers=-1 (multithread bit-identical): {'PASS' if we else 'FAIL'}")
    print("\nALL TESTS", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
