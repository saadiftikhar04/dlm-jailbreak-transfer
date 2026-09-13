"""T02.1: encode the 913-prompt pool with all-mpnet-base-v2 on the HPC.
Reads pool_913.csv (prompt_idx, dataset, original_prompt) and writes:
  mpnet_index.csv  (prompt_idx, dataset, original_prompt -- row alignment)
  mpnet_913.npy    (913 x 768 unit-normalized embeddings, row-aligned)
Run with the raven conda env on a compute node (CPU is fine; no GPU needed).
Recode is deterministic given the same model+input order, so this fully
regenerates the index that build_dedup_mpnet.py requires.
"""
import os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
OUT = os.path.dirname(__file__)

def main(pool_path):
    df = pd.read_csv(pool_path)
    assert len(df) == 913, len(df)
    # keep original CSV order as the canonical row order
    prompts = df["original_prompt"].astype(str).str.strip().fillna("")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    emb = model.encode(prompts.tolist(), batch_size=64, show_progress_bar=True,
                       convert_to_numpy=True, normalize_embeddings=True)
    emb = np.asarray(emb, dtype=np.float32)
    assert emb.shape == (913, 768), emb.shape

    # row alignment file (also serves as mpnet_index.csv)
    idx = df[["prompt_idx", "dataset"]].copy()
    idx["original_prompt"] = prompts
    idx.to_csv(os.path.join(OUT, "mpnet_index.csv"), index=False)
    np.save(os.path.join(OUT, "mpnet_913.npy"), emb)

    # sanity: embed distance of duplicate rows
    print("pool rows:", len(df))
    print("embedding shape:", emb.shape)
    print("norm check (unit, should be ~1):", float(np.linalg.norm(emb[0])))
    print("wrote mpnet_index.csv + mpnet_913.npy")

if __name__ == "__main__":
    pool_path = os.path.join(OUT, "pool_913.csv")
    if len(sys.argv) > 1:
        pool_path = sys.argv[1]
    main(pool_path)