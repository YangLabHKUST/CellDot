"""CellDot — reference-guided optimal-transport decontamination for imaging-based spatial transcriptomics.

Every detected molecule is kept in its host cell, moved to a nearby cell that better explains it, or
removed as background, by solving a sparse capacitated entropic optimal transport problem (Sinkhorn)
against a paired single-cell reference prior and data-calibrated capacity constraints. No neural
network, no training.

Pipeline (raw outs + reference + labels -> cleaned data):
    from celldot import CellDotConfig, clean
    cfg = CellDotConfig(outs="…/outs", reference="…/ref.h5ad", labels="…/labels.parquet",
                        out="…/run", ref_label="celltype", dataset="my_sample")
    clean(cfg)                      # writes cfg.out/{cleaned.h5ad, molecules.parquet, rho_tilde*, ambient, meta}

or on the command line:
    python -m celldot --outs …/outs --reference …/ref.h5ad --labels …/labels.parquet --out …/run

``prep(cfg)`` builds the data contract; ``run(cfg)`` solves and writes the outputs; ``clean(cfg)`` does both.
Labels (cell -> type) are produced by an upstream annotation step (e.g. scANVI) and supplied as a parquet.

Cell identity: every output names a cell by its ORIGINAL platform ``cell_id`` — ``cleaned.h5ad`` is indexed
by it (``obs_names``) and ``molecules.parquet`` records each molecule's source and destination cell by it.
"""
from .config import CellDotConfig

# The solver stack (torch, anndata, scanpy) is imported lazily, so that light tools such as the viewer
# (``celldot-view``) and the converter can be used without it.
_LAZY = {
    "prep": ("prep", "prep"), "run": ("run", "run"), "read_provenance": ("run", "read_provenance"),
    "engine": ("engine", None), "sinkhorn_solve": ("engine", "sinkhorn_solve"),
    "neighborhood_decode": ("engine", "neighborhood_decode"), "estimate_fano": ("engine", "estimate_fano"),
    "background_params": ("engine", "background_params"),
}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        mod, attr = _LAZY[name]; m = importlib.import_module("." + mod, __name__)
        return m if attr is None else getattr(m, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def clean(cfg, do_prep=True):
    """Full pipeline: build the data contract (prep) then solve (run). Returns the cleaned AnnData.

    Set ``do_prep=False`` to reuse artefacts already in ``cfg.out`` and run only the solver."""
    from .prep import prep
    from .run import run
    if do_prep:
        prep(cfg)
    return run(cfg)


__all__ = [
    "CellDotConfig", "prep", "run", "clean", "engine", "read_provenance",
    "sinkhorn_solve", "neighborhood_decode", "estimate_fano", "background_params",
]
__version__ = "0.1.0"
