"""CellDot viewer — an interactive, Xenium-Explorer-style browser view of a CellDot run.

    celldot-view --run <out_dir> --boundaries <cell_boundaries.parquet>

``prep.build_bundle`` turns a result (cleaned.h5ad + transcripts.parquet) plus the platform's cell boundaries into a
spatially sorted, DuckDB-queryable bundle; ``server.create_app`` serves it to the deck.gl front end in ``web/``.
Only what is in the viewport is ever sent to the browser, so the viewer scales from a targeted panel to a
whole-transcriptome section with a billion molecules.
"""
from .prep import build_bundle
from .server import create_app
__all__ = ["build_bundle", "create_app"]
