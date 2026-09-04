"""CellDotConfig — the single config object for a CellDot run.

Holds the input paths (raw Xenium ``outs/`` + reference h5ad + precomputed cell-type ``labels``), the
output directory, and the operating point. All derived artefact paths live under ``out`` so a run is
fully self-contained. Annotation is upstream: ``labels`` is a parquet of (cell_id, celltype[, prob]).

Cell identity: every artefact CellDot writes names a cell by its ORIGINAL ``cell_id`` (the string in the
platform's ``cells.parquet`` / ``transcripts.parquet``). There is no separate integer row id.
"""
import os
from dataclasses import dataclass


@dataclass
class CellDotConfig:
    # ---- inputs ----
    outs: str                       # raw Xenium output dir (transcripts.parquet, cells.parquet, cell_feature_matrix.h5)
    reference: str                  # scRNA reference h5ad (var_names = genes; obs[ref_label] = cell-type vocabulary)
    labels: str                     # parquet of (cell_id, celltype[, prob]) — precomputed annotation (e.g. scANVI)
    out: str                        # output directory (all artefacts written here)
    dataset: str = "dataset"
    ref_label: str = "celltype"     # obs column in the reference giving the type vocabulary
    # ---- file names inside outs (Xenium defaults) ----
    tx_name: str = "transcripts.parquet"
    cells_name: str = "cells.parquet"
    cellfeat_name: str = "cell_feature_matrix.h5"
    unassigned: tuple = ("UNASSIGNED", "-1")   # extracellular sentinel(s) in transcripts.cell_id
    # ---- prep params ----
    qv: float = 20.0                # min Phred-style quality to keep a transcript
    lam: float = 0.1               # ambient profile shrink toward the panel mean (a_g, scoring only)
    min_ref: int = 20              # min reference cells to keep a type
    min_spatial: int = 50          # min spatial cells to keep a type
    gamma_nmin: int = 50           # min observed/expected counts to trust a platform factor (else gamma=1)
    min_prob: float = 0.0          # min label confidence to keep a cell (if labels has a 'prob' column)
    # ---- operating point (the shipped default; used unchanged for every dataset in the paper) ----
    K: int = 14                    # candidate cells per molecule (KDTree neighbours)
    R: float = 15.0                # max assignment distance (um)
    ELL: float = 4.7               # distance length scale in the -d/ELL affinity term (um)
    Z: float = 2.0                 # NB over-dispersed band width (expression capacity)
    PW: float = 0.3                # column-step strength (de-pile hardness)
    KAPPA: float = 1.2             # background capture efficiency: mu_bg = kappa * lambda0[g] * area
    PW_BG: float = 1.0             # firmness of the background-column projection onto mu_bg (1.0 = firm one-sided)
    DROP_DECODE: str = "leak"      # background hard-decode for the non-neighborhood bg_scale modes: 'leak' | 'argmax'
    BG_SCALE: str = "neighborhood"  # background budget granularity. DEFAULT 'neighborhood': per-cell-cap Sinkhorn (Pass 1) + a SLIDING fixed-radius window drop decode (Pass 2, engine.neighborhood_decode). Alternatives kept for ablation: 'metacell_typed' | 'metacell' | 'percell'
    BG_ALPHA: float = 0.12         # a host type is an 'expresser' of gene g (own protected sink) iff rho~[t,g] >= BG_ALPHA * max_t rho~[t,g]; else pooled into the shared POOL sink
    BG_SOFT: float = 0.0           # metacell_typed knob: soft genuine-marker bg-repel exponent (0 = off)
    BG_RADIUS: float = 100.0       # bg_scale='neighborhood': sliding-window disk RADIUS (um) over which the per-(type|POOL,gene) physical drop budget is pooled
    BG_PIX: float = 10.0           # bg_scale='neighborhood': grid resolution (um) for the window-sum convolution
    BG_DEMAND: str = "mass"        # bg_scale='neighborhood' drop-rate denominator: 'mass' (sum of per-molecule background responsibility; DEFAULT) | 'count'
    BG_DILATE: float = 3.0         # post-dilation background (um): excludes UNASSIGNED molecules within this distance of a cell before estimating the background. APPLIED ONLY ON XENIUM 2.0 exports (prep auto-detects); 0 = always raw soup
    EPS: float = 1.0               # entropic Sinkhorn temperature
    NITER: int = 200               # Sinkhorn passes
    TILE: float = 500.0            # tile size (um)
    HALO: float = 15.0             # tile overlap halo (um)
    fano_lo: float = 1.0           # within-type Fano clip lower
    fano_hi: float = 50.0          # within-type Fano clip upper

    # ---- optional explicit artefact paths (override the out-based defaults) ----
    rho_path: str = None
    rho_corrected_path: str = None
    ambient_path: str = None
    meta_path: str = None
    cells_index_path: str = None
    assign_path: str = None
    cleaned_path: str = None
    molecules_path: str = None

    # ---- input paths ----
    @property
    def TX(self): return os.path.join(self.outs, self.tx_name)
    @property
    def CELLS(self): return os.path.join(self.outs, self.cells_name)
    @property
    def CELLFEAT(self): return os.path.join(self.outs, self.cellfeat_name)

    # ---- output artefacts (override path wins, else out-based default) ----
    @property
    def rho(self): return self.rho_path or os.path.join(self.out, "rho_tilde.parquet")              # uncorrected reference prior
    @property
    def rho_corrected(self): return self.rho_corrected_path or os.path.join(self.out, "rho_tilde_corrected.parquet")  # gamma-corrected (model prior)
    @property
    def cells_index(self): return self.cells_index_path or os.path.join(self.out, "cells_index.parquet")  # cell_id, type, centroid (the cell table)
    @property
    def assign_dir(self): return self.assign_path or os.path.join(self.out, "assign")               # per-row-group transcript shards
    @property
    def ambient(self): return self.ambient_path or os.path.join(self.out, "ambient_profile_ag.parquet")  # gene -> n_extra, a_g
    @property
    def meta(self): return self.meta_path or os.path.join(self.out, "dataset_meta.json")
    @property
    def cleaned(self): return self.cleaned_path or os.path.join(self.out, "cleaned.h5ad")           # layers raw/greedy/celldot
    @property
    def molecules(self): return self.molecules_path or os.path.join(self.out, "molecules.parquet")  # per-molecule fate
