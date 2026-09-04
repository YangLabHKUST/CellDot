# CellDot

**Molecule-level decontamination for imaging-based spatial transcriptomics.**

Imaging-based platforms (Xenium, MERSCOPE, CosMx, and whole-transcriptome panels such as Atera) detect
individual RNA molecules, but segmentation errors, transcript spillover and three-dimensional cell overlap
assign many of them to the wrong cell. CellDot decides the fate of **every transcript** by solving one
optimal-transport problem over the whole section:

| fate | meaning |
|---|---|
| **keep** | the molecule stays in the cell it was assigned to |
| **move** | it is reassigned to a nearby cell whose type explains it better |
| **drop** | it is removed as background |

The decision combines a paired single-cell reference (which genes each cell type expresses), spatial
distance, a per-cell *expression capacity* (a cell cannot hold more of a gene than its type is expected to
express) and a per-cell *background capacity* measured from the extracellular molecules of the same
section. There is no training and nothing to tune: one operating point was used for every dataset in the
paper. The output is a corrected cell × gene matrix **and** a traceable fate for each molecule, which the
bundled viewer lets you inspect cell by cell.

---

## Installation

```bash
git clone https://github.com/YangLabHKUST/CellDot.git
cd CellDot
pip install -e .              # core: numpy scipy pandas torch anndata pyarrow scikit-learn scanpy
pip install -e ".[viewer]"    # + duckdb fastapi uvicorn h5py  (the interactive viewer)
pip install scikit-image      # only for Xenium 2.0 exports (post-dilation background estimate)
```

Python ≥ 3.10. A CUDA GPU is used automatically when available; otherwise CellDot runs on the CPU
(slower, same result).

## Quick start

```bash
celldot --outs  /data/sample/outs \
        --reference /data/sample/reference.h5ad \
        --labels /data/sample/labels.parquet \
        --out   /data/sample/celldot \
        --ref-label celltype --dataset sample

celldot-view --run /data/sample/celldot --boundaries /data/sample/outs/cell_boundaries.parquet
```

The same from Python:

```python
from celldot import CellDotConfig, clean
cfg = CellDotConfig(outs="/data/sample/outs", reference="/data/sample/reference.h5ad",
                    labels="/data/sample/labels.parquet", out="/data/sample/celldot",
                    ref_label="celltype", dataset="sample")
adata = clean(cfg)          # prep (reference prior, background estimate) + solve
```

`clean` runs the two stages `prep` and `run`; `clean(cfg, do_prep=False)` (or `celldot --skip-prep`)
re-solves with the prep artefacts already in the output directory.

## Inputs

| input | what CellDot needs |
|---|---|
| `outs/` | the platform export: `transcripts.parquet` (`feature_name`, `x_location`, `y_location`, `cell_id`, `qv`), `cells.parquet` (`cell_id`, `x_centroid`, `y_centroid`, `cell_area`, …) and `cell_feature_matrix.h5` (10x HDF5; used for the gene panel). These are the standard Xenium output files. Other platforms work as long as the same columns are provided; `--unassigned` sentinels default to `UNASSIGNED` and `-1`. |
| reference `.h5ad` | a single-cell reference of the same tissue with **raw counts** in `X` and the cell-type label in `obs[ref_label]`. Genes are matched by `var_names`; only panel genes present in the reference are used. |
| `labels.parquet` | the spatial cells' cell types: columns `cell_id`, `celltype` (and optionally `prob`). The vocabulary must be the reference's. In the paper the labels come from scANVI label transfer; any annotation works. Cells without a label are left untouched. |

For Xenium 2.0 exports (multimodal segmentation; detected automatically from `cells.parquet`), CellDot
excludes unassigned molecules within 3 µm of a cell before estimating the background (`BG_DILATE`).

## Outputs

Everything is written to `--out`. **Cells are identified only by their original `cell_id`**, exactly as in
`cells.parquet` and `transcripts.parquet` (stored as strings). There is no separate integer cell index.

### `cleaned.h5ad`

```python
import anndata as ad
A = ad.read_h5ad("celldot/cleaned.h5ad")
A.obs_names            # the original cell ids
A.layers["celldot"]    # corrected counts  (A.layers["raw"] = before; A.layers["greedy"] = no-capacity baseline)
A.obs[["type", "x_centroid", "y_centroid", "n_dropped", "n_moved_out", "n_moved_in", "drop_frac", "mu_bg"]]
A.var["lambda0"]       # per-gene background density (molecules / µm²)
A.uns["celldot"]       # provenance: version, parameters, run_id, fate totals
```

### `molecules.parquet`

One row per molecule that entered the run (quality ≥ 20, panel gene, assigned to a labelled cell):

| column | content |
|---|---|
| `x`, `y` | position (µm) |
| `gene` | gene name |
| `old_host` | the cell the platform assigned the molecule to (`cell_id`) |
| `new_host` | the cell it belongs to after correction (`cell_id`); **empty string** when removed as background |
| `action` | `keep`, `move` or `drop` |

String columns are dictionary-encoded, so the file stays compact even for a billion molecules.

```python
import pandas as pd
m = pd.read_parquet("celldot/molecules.parquet")
m.action.value_counts()
moved = m[m.action == "move"]                       # every reassigned molecule: from old_host to new_host
m[m.old_host == "aaabbccd-1"]                       # the fate of one cell's molecules
```

The corrected matrix is exactly the kept molecules counted by `new_host` × `gene`, and
`celldot.read_provenance(path)` returns the same provenance record from either file (a matched pair shares
`run_id`).

### Other files

`rho_tilde.parquet` / `rho_tilde_corrected.parquet` (reference prior and its platform-corrected form),
`ambient_profile_ag.parquet` and `dataset_meta.json` (background estimate), `cells_index.parquet` and
`assign/` (the cell table and per-molecule shards written by `prep`).

## The viewer

```bash
celldot-view --run <out dir> --boundaries <cell_boundaries.parquet>   # then open http://127.0.0.1:8765
```

An interactive, Xenium-Explorer-style map of the section built on deck.gl, with a small DuckDB server
behind it. Only what is in the viewport is ever sent to the browser, so it works the same on a 300-gene
panel and on a whole-transcriptome section with a billion molecules.

- **Cells** as polygons (when zoomed in) or points (whole section), coloured by cell type, by the raw or
  cleaned expression of a gene, by the *change* (cleaned − raw), or by the fraction of molecules dropped
  or moved out. Click a type in the legend to hide it.
- **Molecules** of the genes you pick, or of *all* genes once you are zoomed in, coloured by fate or by
  gene, with arrows from each moved molecule to the cell it now belongs to. Hover for gene, fate, source
  and destination cell.
- **Click a cell** to see its id, type, molecule counts, every gene's raw and cleaned count, and all of
  its molecules highlighted: kept, arriving, leaving and dropped.
- **Find** a cell by id or jump to coordinates; the URL keeps the view and the selected genes, so it can
  be shared; save the current view as PNG; the status bar reports fate fractions inside the view.

The first launch builds a query bundle next to the run (`<out>/viewer_bundle/`, roughly the size of the
molecule table; a minute per hundred million molecules). Use `--rebuild` after re-running CellDot,
`--port` to change the port, `--no-browser` on a server (then tunnel the port with ssh).

## Results from the research package (`spdenoise`)

CellDot is the release form of the `spdenoise` research code and gives **exactly** the same result; only
the identity scheme changed (spdenoise numbered cells and genes by position). Convert an old result so the
viewer and other CellDot tools can read it:

```bash
celldot-convert --h5ad bench.h5ad --molecules molecules.parquet --out <dir> [--cells-index cells_index.parquet]
```

`tests/compare_outputs.py --celldot <dir> --spdenoise <dir>` checks two runs against each other.

## Parameters

All defaults live in `CellDotConfig`; they are the shipped operating point and were used unchanged for
every dataset in the paper. They rarely need to change.

| field | default | role |
|---|---|---|
| `K`, `R` | 14, 15 µm | candidate cells per molecule and the maximum reassignment distance |
| `ELL` | 4.7 µm | distance scale of the transport cost |
| `Z` | 2 | width of the expression-capacity band |
| `KAPPA` | 1.2 | background capture efficiency (background capacity = κ · λ⁰ · cell area) |
| `PW`, `PW_BG` | 0.3, 1.0 | firmness of the two capacity projections |
| `EPS`, `NITER` | 1, 200 | entropic temperature and Sinkhorn sweeps |
| `TILE`, `HALO` | 500 µm, 15 µm | tiling of the section (lower `TILE` to reduce memory) |
| `BG_RADIUS` | 100 µm | window over which the background budget is pooled when decoding drops |
| `BG_DILATE` | 3 µm | margin excluded from the background estimate on Xenium 2.0 exports |
| `qv`, `min_ref`, `min_spatial` | 20, 20, 50 | molecule quality cut; minimum reference / spatial cells per type |

CLI overrides: `--qv --ell --z --pw --kappa --niter`.

## Resources

Runtime and memory scale with the number of molecules. On one V100 GPU the breast cancer section
(168 k cells, 31 M molecules) takes about five minutes; a whole-transcriptome section (718 k cells, 881 M
molecules, 17 k genes) takes a few hours and several hundred GB of RAM. The solver is tiled, so GPU
memory needs are modest.

## Tests

```bash
python tests/test_neighborhood_decode.py    # window decode against a brute-force reference
python tests/test_equivalence.py            # synthetic Xenium-style data: output contract, CLI, viewer
                                            # inputs, and exact equality with spdenoise if present
```

## Layout

| path | what |
|---|---|
| `celldot/engine.py` | the solver (Sinkhorn with capacities, window decode, background estimators) |
| `celldot/prep.py`, `celldot/run.py` | the two stages; `config.py` holds `CellDotConfig` |
| `celldot/viewer/` | `celldot-view`: bundle builder, DuckDB API server, deck.gl front end |
| `celldot/convert.py` | `celldot-convert` |
| `tests/` | unit and end-to-end tests, output comparison, synthetic data generator |

## Citation

Chen Y., Liu Y., et al. *Accurate and scalable decontamination of imaging-based spatial transcriptomics
via optimal transport* (manuscript in preparation).
