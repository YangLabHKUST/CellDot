"""CLI entry point:  python -m celldot --outs … --reference … --labels … --out …

Runs the full pipeline (prep + solve) and writes cleaned.h5ad + molecules.parquet (+ priors/ambient/meta)
into --out. Operating-point flags override the CellDotConfig defaults; --skip-prep reuses existing artefacts.
"""
import argparse, os
from .config import CellDotConfig
from . import clean


def main():
    ap = argparse.ArgumentParser(
        prog="celldot",
        description="reference-guided optimal-transport decontamination -> cleaned.h5ad + per-molecule fates")
    ap.add_argument("--outs", required=True, help="raw Xenium output dir (transcripts.parquet, cells.parquet, cell_feature_matrix.h5)")
    ap.add_argument("--reference", required=True, help="scRNA reference h5ad (obs[ref-label] = cell types)")
    ap.add_argument("--labels", required=True, help="labels.parquet (cell_id, celltype[, prob]) from upstream annotation")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--ref-label", default="celltype", help="reference obs column with the type vocabulary")
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--skip-prep", action="store_true", help="reuse artefacts already in --out (run solver only)")
    # operating-point overrides (default = CellDotConfig defaults)
    ap.add_argument("--qv", type=float); ap.add_argument("--ell", type=float); ap.add_argument("--z", type=float)
    ap.add_argument("--pw", type=float); ap.add_argument("--kappa", type=float); ap.add_argument("--niter", type=int)
    args = ap.parse_args()

    kw = dict(outs=args.outs, reference=args.reference, labels=args.labels, out=args.out,
              ref_label=args.ref_label, dataset=args.dataset)
    for flag, field in [("qv", "qv"), ("ell", "ELL"), ("z", "Z"), ("pw", "PW"), ("kappa", "KAPPA"), ("niter", "NITER")]:
        v = getattr(args, flag, None)
        if v is not None: kw[field] = v
    os.makedirs(args.out, exist_ok=True)
    clean(CellDotConfig(**kw), do_prep=not args.skip_prep)


if __name__ == "__main__":
    main()
