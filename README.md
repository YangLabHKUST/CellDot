<p align="center"><img src="docs/celldot-lockup.svg" width="380" alt="CellDot"></p>

<p align="center"><b>Every molecule in the right cell.</b><br>
Molecule-level decontamination for imaging-based spatial transcriptomics (Xenium, MERSCOPE, CosMx, whole-transcriptome panels).</p>

<p align="center"><a href="https://viewer.celldot.online">Live demo</a> · <a href="#install">Install</a> · <a href="#run-celldot-in-three-steps">Run</a> · <a href="#outputs">Outputs</a> · <a href="#the-viewer">Viewer</a></p>

---

Segmentation errors, spillover and 3-D overlap put many detected molecules in the wrong cell. CellDot looks at
**every molecule** of a section and decides, with one optimal-transport problem over the whole tissue:

| fate | |
|:--|:--|
| 🩶 **keep** | the molecule stays in its cell |
| 🔵 **move** | it belongs to a neighbouring cell and is reassigned there |
| 🔴 **drop** | it is ambient background and is removed |

The decision uses a single-cell reference of the tissue, the distance to nearby cells, how much of each gene a
cell of that type can hold, and the background measured in the section itself. Nothing to train, nothing to tune:
one setting was used for every dataset in the paper. You get a corrected cell × gene matrix **and** the fate of
each molecule, which the bundled viewer shows on the tissue.

<p align="center"><a href="https://viewer.celldot.online/d/CRC/"><img src="docs/viewer.jpg" width="820" alt="CellDot viewer: a colorectal cancer section, CLCA1 molecules coloured by fate"></a></p>

## Install

```bash
git clone https://github.com/YangLabHKUST/CellDot.git
cd CellDot
pip install -e ".[viewer,annotate]"
```

Python ≥ 3.10. A CUDA GPU is used when present; the CPU gives the same result, more slowly.

## Run CellDot in three steps

**1. Cell types for the spatial cells.** CellDot needs a label for each cell in the reference's vocabulary. If you
do not have one yet, `celldot-annotate` transfers the reference labels with scANVI (the recipe used in the paper);
any other annotation works too (marker scoring, cell2location, Tangram, …) as long as it yields a table with
`cell_id` and `celltype`.

```bash
celldot-annotate --input outs/ --reference reference.h5ad --output labels.parquet
```

**2. Clean.**

```bash
celldot --input outs/ --reference reference.h5ad --labels labels.parquet --output celldot/
```

**3. Look at the result.**

```bash
celldot-view --run celldot/ --boundaries outs/cell_boundaries.parquet      # opens http://127.0.0.1:8765
```

`outs/` is the folder your platform exported (for Xenium, the `outs` folder with `transcripts.parquet`,
`cells.parquet`, `cell_feature_matrix.h5` and `cell_boundaries.parquet`). The breast cancer section of the paper
(168 k cells, 31 M molecules) takes about five minutes on one GPU.

## Inputs

| | |
|:--|:--|
| `--input` | the platform's output folder: `transcripts.parquet`, `cells.parquet`, `cell_feature_matrix.h5` |
| `--reference` | a single-cell reference of the same tissue (`.h5ad`, raw counts, cell type in `obs["celltype"]`; another column with `--ref-label-col`) |
| `--labels` | the spatial cells' types: a parquet with `cell_id`, `celltype` (step 1) |

## Outputs

Two files in `--output`, and nothing else you need to read:

| | |
|:--|:--|
| `cleaned.h5ad` | the cells. `X` is the corrected count matrix, `layers["raw"]` the counts before, `obs["type"]` the cell type; `obs_names` are the original cell ids. |
| `transcripts.parquet` | the molecules: your input table, row for row and every column unchanged, plus `celldot_fate` (`keep` / `move` / `drop`) and `celldot_cell_id` (the cell the molecule belongs to now). |

```python
import anndata as ad, pandas as pd
A = ad.read_h5ad("celldot/cleaned.h5ad")          # A.X corrected, A.layers["raw"] before
t = pd.read_parquet("celldot/transcripts.parquet")
t.celldot_fate.value_counts()                     # keep / move / drop (+ background, not_evaluated: untouched rows)
```

Molecules that were extracellular in the input (`background`) or not evaluated (low quality, gene outside the
panel, unlabelled cell) keep their original assignment.

## The viewer

`celldot-view` is an interactive map of the section in your browser: cells coloured by type or by the expression
of a gene before and after correction, the molecules of the genes you pick coloured by fate, an arrow from every
moved molecule to its new cell, and a click on any cell lists all of its molecules. Try it on the paper's datasets
at **[viewer.celldot.online](https://viewer.celldot.online)**.

The first launch builds a query index next to the result (a minute per hundred million molecules); later launches
are instant. To open on a chosen scene, press `shift+D` in the viewer: the current position, genes and settings
are saved next to `cleaned.h5ad` and used from then on.

## Python

```python
from celldot import CellDotConfig, clean
A = clean(CellDotConfig(input="outs/", reference="reference.h5ad", labels="labels.parquet", output="celldot/"))
```

`celldot --help` lists the advanced parameters; they were left at their defaults for every dataset in the paper.

## Citation

Chen Y., Liu Y., et al. *Accurate and scalable decontamination of imaging-based spatial transcriptomics via optimal
transport.* (manuscript in preparation)
