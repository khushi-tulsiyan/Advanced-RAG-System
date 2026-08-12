#!/usr/bin/env python3
"""Chunks the raw corpus and builds the hybrid retrieval index.

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --input-dir data/raw_docs --chunk-size 800
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.chunker import TextChunker  # noqa: E402
from rag.config import settings as default_settings  # noqa: E402
from rag.embeddings import load_embedder  # noqa: E402
from rag.retriever import HybridRetriever  # noqa: E402

logger = logging.getLogger("build_index")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=None, help="Directory of .txt/.md documents")
    parser.add_argument("--data-dir", type=Path, default=None, help="Root data directory for outputs")
    parser.add_argument("--chunk-size", type=int, default=None, help="Characters per chunk")
    parser.add_argument("--chunk-overlap", type=int, default=None, help="Character overlap between chunks")
    parser.add_argument("--embedding-model", type=str, default=None, help="sentence-transformers model name")
    parser.add_argument("--no-faiss", action="store_true", help="Force the numpy vector backend")
    parser.add_argument(
        "--save-chunk-files",
        action="store_true",
        help="Also write per-document chunk JSON files to data/processed_chunks",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    overrides = {}
    if args.data_dir:
        overrides["data_dir"] = args.data_dir.resolve()
    if args.chunk_size:
        overrides["chunk_size"] = args.chunk_size
    if args.chunk_overlap is not None:
        overrides["chunk_overlap"] = args.chunk_overlap
    if args.embedding_model:
        overrides["embedding_model"] = args.embedding_model
    settings = replace(default_settings, **overrides) if overrides else default_settings
    settings.ensure_dirs()

    input_dir = args.input_dir or settings.raw_docs_dir
    chunker = TextChunker(settings=settings)

    logger.info("Chunking documents from %s", input_dir)
    if args.save_chunk_files:
        chunks = chunker.process_documents(input_dir, settings.processed_chunks_dir)
    else:
        chunks = chunker.chunk_directory(input_dir)

    if not chunks:
        logger.error("No chunks produced. Add .txt or .md files to %s and retry.", input_dir)
        return 1

    logger.info("Embedding %d chunks with %s", len(chunks), settings.embedding_model)
    embedder = load_embedder(settings.embedding_model)
    retriever = HybridRetriever.from_documents(
        chunks, embedder=embedder, settings=settings, prefer_faiss=not args.no_faiss
    )

    retriever.save(settings)
    logger.info(
        "Indexed %d chunks from %d documents -> %s",
        len(chunks),
        len({chunk.source for chunk in chunks}),
        settings.vector_store_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
