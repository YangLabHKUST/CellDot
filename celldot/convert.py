"""celldot-convert: bring an spdenoise (v0.1.x) result into the CellDot output format.

spdenoise named cells by an integer row (obs column 'cell_id' / cells_index.parquet; molecules old_host/new_host = row,
-1 = dropped; gene = var index). This rewrites
    cleaned.h5ad       obs indexed by cell_id (no cell_id column), layer 'spdenoise' -> 'celldot', uns['spd'] -> uns['celldot']
    molecules.parquet  gene -> gene name, old_host/new_host -> cell_id ("" = dropped), dictionary-encoded, streamed per row group
so that CellDot tools (the viewer included) can read it.

    celldot-convert --h5ad bench.h5ad --molecules molecules.parquet --out <dir> [--cells-index cells_index.parquet]
"""
import argparse, json, os, numpy as np, pandas as pd
import pyarrow as pa, pyarrow.parquet as pq


def convert(h5ad, molecules, out, cells_index=None, log=print):
    import anndata as ad
    os.makedirs(out, exist_ok=True)
    A = ad.read_h5ad(h5ad)
    if cells_index:
        ci = pd.read_parquet(cells_index).sort_values("row"); cell_ids = ci["cell_id"].astype(str).values
        assert len(cell_ids) == A.n_obs
    elif "cell_id" in A.obs.columns:
        cell_ids = A.obs["cell_id"].astype(str).values
    else:
        cell_ids = A.obs_names.astype(str).values
    assert len(set(cell_ids)) == len(cell_ids) and "" not in set(cell_ids), "cell ids must be unique and non-empty"
    genes = np.asarray(A.var_names, dtype=object)
    A.obs.index = pd.Index(cell_ids, name="cell_id")
    if "cell_id" in A.obs.columns: A.obs = A.obs.drop(columns=["cell_id"])
    if "spdenoise" in A.layers and "celldot" not in A.layers:
        A.layers["celldot"] = A.layers["spdenoise"]; del A.layers["spdenoise"]
    prov = dict(A.uns.get("spd", A.uns.get("celldot", {}))); prov["converted_from"] = "spdenoise"; prov["id_scheme"] = "cell_id"
    if "spd" in A.uns: del A.uns["spd"]
    A.uns["celldot"] = prov
    A.write_h5ad(os.path.join(out, "cleaned.h5ad")); log(f"wrote cleaned.h5ad ({A.n_obs:,} cells x {A.n_vars:,} genes, obs_names = cell_id)")

    pf = pq.ParquetFile(molecules); sch = pf.schema_arrow
    if str(sch.field("old_host").type).startswith("dictionary") or sch.field("old_host").type == pa.string():
        log("molecules.parquet already uses cell_id strings; copying"); import shutil; shutil.copy(molecules, os.path.join(out, "molecules.parquet")); return
    host_dict = pa.array([""] + list(cell_ids), pa.string()); gene_dict = pa.array(list(genes), pa.string()); act_dict = pa.array(["keep", "move", "drop"], pa.string())
    md = dict(sch.metadata or {}); md[b"celldot"] = json.dumps(prov, default=str).encode(); md.pop(b"spd", None)
    out_schema = pa.schema([("x", pa.float32()), ("y", pa.float32()), ("gene", pa.dictionary(pa.int16() if len(genes) < 32000 else pa.int32(), pa.string())),
                            ("old_host", pa.dictionary(pa.int32(), pa.string())), ("new_host", pa.dictionary(pa.int32(), pa.string())),
                            ("action", pa.dictionary(pa.int8(), pa.string()))], metadata=md)
    w = pq.ParquetWriter(os.path.join(out, "molecules.parquet"), out_schema, compression="zstd"); n = 0
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=["x", "y", "gene", "old_host", "new_host", "action"])
        oh = t.column("old_host").to_numpy().astype(np.int32); nh = t.column("new_host").to_numpy().astype(np.int32)
        g = t.column("gene").to_numpy().astype(np.int16 if len(genes) < 32000 else np.int32)
        act = t.column("action").to_pandas().map({"keep": 0, "move": 1, "drop": 2}).values.astype(np.int8)
        batch = pa.table({"x": t.column("x").cast(pa.float32()), "y": t.column("y").cast(pa.float32()),
                          "gene": pa.DictionaryArray.from_arrays(pa.array(g), gene_dict),
                          "old_host": pa.DictionaryArray.from_arrays(pa.array(oh + 1), host_dict),
                          "new_host": pa.DictionaryArray.from_arrays(pa.array(nh + 1), host_dict),
                          "action": pa.DictionaryArray.from_arrays(pa.array(act), act_dict)}, schema=out_schema)
        w.write_table(batch); n += t.num_rows
        if rg % 20 == 0 or rg == pf.num_row_groups - 1: log(f"  molecules row group {rg + 1}/{pf.num_row_groups}  {n:,} rows")
    w.close(); log(f"wrote molecules.parquet ({n:,} molecules; hosts as cell_id, gene as name)")


def main():
    ap = argparse.ArgumentParser(prog="celldot-convert", description="convert an spdenoise result to the CellDot output format")
    ap.add_argument("--h5ad", required=True); ap.add_argument("--molecules", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--cells-index", help="cells_index.parquet (row -> cell_id); default: the h5ad obs 'cell_id' column")
    a = ap.parse_args(); convert(a.h5ad, a.molecules, a.out, a.cells_index)


if __name__ == "__main__":
    main()
