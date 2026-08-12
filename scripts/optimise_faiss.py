#!/usr/bin/env python3
"""Compresses the flat FAISS index into an IVF-PQ index for large corpora.

Fixes over the original version, which trained the quantiser on
``np.random.rand`` (the codebook then describes noise, not the corpus),
hard-coded ``nlist=100`` regardless of corpus size, and assumed ``m=8`` divides
the embedding dimension.

IVF-PQ is approximate: it trades recall for memory. This script measures that
trade-off against the exact index and refuses to run on corpora too small to
train a meaningful codebook.

Usage:
    python scripts/optimise_faiss.py --nlist 128 --m 16
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.config import settings  # noqa: E402

logger = logging.getLogger("optimise_faiss")

# FAISS warns below 39 training points per centroid and errors below ~30.
MIN_POINTS_PER_CENTROID = 39


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nlist", type=int, default=None, help="Number of IVF cells (default: ~sqrt(N))")
    parser.add_argument("--m", type=int, default=None, help="PQ sub-quantisers (must divide the dimension)")
    parser.add_argument("--nbits", type=int, default=8, help="Bits per PQ sub-quantiser")
    parser.add_argument("--nprobe", type=int, default=8, help="Cells probed at search time")
    parser.add_argument("--eval-queries", type=int, default=200, help="Sample size for the recall check")
    return parser.parse_args()


def largest_divisor_at_most(dimension: int, limit: int) -> int:
    """Largest ``m <= limit`` that divides ``dimension``; PQ requires an exact split."""
    for candidate in range(min(limit, dimension), 0, -1):
        if dimension % candidate == 0:
            return candidate
    return 1


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    try:
        import faiss
    except ImportError:
        logger.error("faiss is not installed. Install it with: pip install faiss-cpu")
        return 1

    source = settings.faiss_index_path
    if not source.exists():
        logger.error("No FAISS index at %s. Run scripts/build_index.py first.", source)
        return 1

    index = faiss.read_index(str(source))
    total, dimension = index.ntotal, index.d
    logger.info("Loaded exact index: %d vectors x %d dims", total, dimension)

    nlist = args.nlist or max(1, min(4096, int(np.sqrt(total))))
    required = nlist * MIN_POINTS_PER_CENTROID
    if total < required:
        logger.error(
            "Corpus too small for IVF-PQ: %d vectors but nlist=%d needs ~%d for training. "
            "The exact IndexFlatIP is already the right choice at this scale.",
            total,
            nlist,
            required,
        )
        return 1

    m = args.m or largest_divisor_at_most(dimension, 16)
    if dimension % m:
        logger.error("m=%d does not divide the embedding dimension %d", m, dimension)
        return 1

    vectors = index.reconstruct_n(0, total).astype(np.float32)

    quantizer = faiss.IndexFlatIP(dimension)
    compressed = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, args.nbits, faiss.METRIC_INNER_PRODUCT)

    logger.info("Training IVF-PQ on the real corpus (nlist=%d, m=%d, nbits=%d)", nlist, m, args.nbits)
    compressed.train(vectors)
    compressed.add(vectors)
    compressed.nprobe = args.nprobe

    target = settings.vector_store_dir / "faiss_ivfpq.index"
    faiss.write_index(compressed, str(target))

    recall = measure_recall(index, compressed, vectors, args.eval_queries)
    exact_bytes = source.stat().st_size
    compressed_bytes = target.stat().st_size
    logger.info(
        "Saved %s | size %.1fMB -> %.1fMB (%.1fx smaller) | recall@10 vs exact: %.3f",
        target,
        exact_bytes / 1e6,
        compressed_bytes / 1e6,
        exact_bytes / max(compressed_bytes, 1),
        recall,
    )
    if recall < 0.8:
        logger.warning("Recall is low. Raise --nprobe or --m, or keep the exact index.")
    logger.info("To use it, replace %s with this file (back up the original first).", source.name)
    return 0


def measure_recall(exact, compressed, vectors: np.ndarray, sample_size: int, k: int = 10) -> float:
    """Fraction of exact top-k neighbours the compressed index still returns."""
    rng = np.random.default_rng(0)
    sample_size = min(sample_size, vectors.shape[0])
    queries = vectors[rng.choice(vectors.shape[0], size=sample_size, replace=False)]

    _, exact_ids = exact.search(queries, k)
    _, approx_ids = compressed.search(queries, k)

    overlap = sum(len(set(a.tolist()) & set(b.tolist())) for a, b in zip(exact_ids, approx_ids))
    return overlap / (sample_size * k)


if __name__ == "__main__":
    raise SystemExit(main())
