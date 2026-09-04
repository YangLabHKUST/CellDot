"""celldot-view: build (once) and serve the interactive viewer of a CellDot run.

    celldot-view --run <out_dir> --boundaries <cell_boundaries.parquet> [--port 8765] [--rebuild]
    celldot-view --run <out_dir> --outs <platform outs dir>          # boundaries = <outs>/cell_boundaries.parquet
"""
import argparse, os, sys, threading, webbrowser


def main():
    ap = argparse.ArgumentParser(prog="celldot-view", description="interactive viewer of a CellDot run (cells, molecules and their fates)")
    ap.add_argument("--run", required=True, help="CellDot output dir (cleaned.h5ad + molecules.parquet)")
    ap.add_argument("--boundaries", help="cell_boundaries.parquet (cell_id, vertex_x, vertex_y)")
    ap.add_argument("--outs", help="platform output dir; boundaries taken from <outs>/cell_boundaries.parquet")
    ap.add_argument("--bundle", help="where to keep the viewer bundle (default <run>/viewer_bundle)")
    ap.add_argument("--rebuild", action="store_true", help="rebuild the bundle even if it exists")
    ap.add_argument("--prep-only", action="store_true", help="build the bundle and exit (no server)")
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--threads", type=int, default=4, help="DuckDB threads"); ap.add_argument("--tile", type=float, default=100.0, help="spatial sort tile (um)")
    ap.add_argument("--memory-limit", default="8GB", help="DuckDB memory limit while building the bundle")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    bundle = a.bundle or os.path.join(a.run, "viewer_bundle")
    if a.rebuild or not os.path.exists(os.path.join(bundle, "meta.json")):
        b = a.boundaries or (os.path.join(a.outs, "cell_boundaries.parquet") if a.outs else None)
        if not b or not os.path.exists(b): sys.exit("need --boundaries <cell_boundaries.parquet> (or --outs <dir>) to build the bundle")
        from .prep import build_bundle
        build_bundle(a.run, b, bundle, tile=a.tile, threads=a.threads, memory_limit=a.memory_limit)
    if a.prep_only: return
    from .server import create_app
    import uvicorn
    app = create_app(bundle, threads=a.threads); m = app.state.meta
    url = f"http://{a.host}:{a.port}"
    print(f"CellDot viewer [{m['dataset']}]  {m['n_cells']:,} cells  {m['n_tx']:,} molecules  {m['n_genes']:,} genes\n  -> {url}", flush=True)
    if not a.no_browser: threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
