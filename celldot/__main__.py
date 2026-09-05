"""CLI entry point:  celldot --input <platform output folder> --reference ref.h5ad --labels labels.parquet --output <folder>

Runs the full pipeline (prep + solve) and writes cleaned.h5ad + transcripts.parquet into --output.
Operating-point flags override the CellDotConfig defaults; --skip-prep reuses the intermediates already there.
"""
import argparse, os, sys
from .config import CellDotConfig
from . import clean


def main():
    ap = argparse.ArgumentParser(prog="celldot", description="CellDot: decide the fate of every molecule (keep / move / drop) -> cleaned.h5ad + transcripts.parquet")
    ap.add_argument("--input", required=True, help="the platform's output folder (transcripts.parquet, cells.parquet, cell_feature_matrix.h5)")
    ap.add_argument("--reference", required=True, help="single-cell reference .h5ad (raw counts; a cell-type column in obs)")
    ap.add_argument("--labels", required=True, help="cell types of the spatial cells: labels.parquet with cell_id, celltype (e.g. from celldot-annotate)")
    ap.add_argument("--output", required=True, help="output folder")
    ap.add_argument("--ref-label-col", default="celltype", help="column of reference.obs with the cell type (default: celltype)")
    ap.add_argument("--ref-counts-layer", help="layer of the reference with the raw counts (default: layers['counts'] if present, else X)")
    ap.add_argument("--name", help="a label stored in the outputs and shown by the viewer (default: the input folder's name)")
    ap.add_argument("--skip-prep", action="store_true", help="reuse the intermediates already in --output (run the solver only)")
    adv = ap.add_argument_group("advanced (the defaults were used for every dataset in the paper)")
    adv.add_argument("--qv", type=float, help="minimum transcript quality (default 20)")
    adv.add_argument("--ell", type=float, help="distance scale of the transport cost, um (default 4.7)")
    adv.add_argument("--z", type=float, help="width of the expression-capacity band (default 2)")
    adv.add_argument("--pw", type=float, help="firmness of the capacity projection (default 0.3)")
    adv.add_argument("--kappa", type=float, help="background capture efficiency (default 1.2)")
    adv.add_argument("--niter", type=int, help="Sinkhorn sweeps (default 200)")
    args = ap.parse_args()

    kw = dict(input=args.input, reference=args.reference, labels=args.labels, output=args.output, ref_label_col=args.ref_label_col,
              ref_counts_layer=args.ref_counts_layer, name=args.name)
    for flag, field in [("qv", "qv"), ("ell", "ELL"), ("z", "Z"), ("pw", "PW"), ("kappa", "KAPPA"), ("niter", "NITER")]:
        v = getattr(args, flag, None)
        if v is not None: kw[field] = v
    try:
        cfg = CellDotConfig(**kw)                                   # refuses --output inside/equal to --input
        os.makedirs(args.output, exist_ok=True)
        clean(cfg, do_prep=not args.skip_prep)
    except (ValueError, ImportError) as e:                          # input problems: one clear line, no traceback
        sys.exit(f"celldot: {e}")


if __name__ == "__main__":
    main()
