from __future__ import annotations

import json

import pytest

from rag.chunker import TextChunker


def test_chunks_respect_size_limit():
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    text = " ".join(f"word{i}" for i in range(400))
    chunks = chunker.chunk_text(text)

    assert chunks
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_short_text_is_a_single_chunk():
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    assert chunker.chunk_text("A short sentence.") == ["A short sentence."]


def test_empty_and_whitespace_text_produce_no_chunks():
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    assert chunker.chunk_text("") == []
    assert chunker.chunk_text("   \n\n  ") == []


def test_chunks_overlap_to_preserve_context_across_boundaries():
    chunker = TextChunker(chunk_size=120, chunk_overlap=40)
    text = ". ".join(f"Sentence number {i} carries some content" for i in range(30))
    chunks = chunker.chunk_text(text)

    assert len(chunks) > 1
    # Consecutive chunks should share text, otherwise a fact split across a
    # boundary becomes unretrievable.
    overlaps = [chunks[i][-20:] in chunks[i + 1] for i in range(len(chunks) - 1)]
    assert any(overlaps)


def test_overlap_does_not_start_mid_word():
    """A tail sliced blindly yields chunks beginning like 'ong at exact term'."""
    chunker = TextChunker(chunk_size=120, chunk_overlap=40)
    text = " ".join(f"distinctiveword{i} filler content here" for i in range(40))
    vocabulary = set(text.split())

    for chunk in chunker.chunk_text(text):
        assert chunk.split()[0] in vocabulary, f"chunk starts mid-word: {chunk[:40]!r}"


def test_sentence_punctuation_survives_splitting():
    chunker = TextChunker(chunk_size=60, chunk_overlap=10)
    text = "First fact here. Second fact here. Third fact here. Fourth fact here."
    joined = " ".join(chunker.chunk_text(text))
    assert "." in joined


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError):
        TextChunker(chunk_size=0, chunk_overlap=0)
    with pytest.raises(ValueError):
        TextChunker(chunk_size=100, chunk_overlap=100)


def test_chunk_document_attaches_provenance():
    chunker = TextChunker(chunk_size=80, chunk_overlap=10)
    chunks = chunker.chunk_document("Some text. " * 40, source="notes.txt")

    assert all(chunk.source == "notes.txt" for chunk in chunks)
    assert [chunk.position for chunk in chunks] == list(range(len(chunks)))
    assert len({chunk.id for chunk in chunks}) == len(chunks), "chunk ids must be unique"


def test_chunk_ids_are_deterministic():
    chunker = TextChunker(chunk_size=80, chunk_overlap=10)
    first = chunker.chunk_document("Repeatable content here.", source="a.txt")
    second = chunker.chunk_document("Repeatable content here.", source="a.txt")
    assert [c.id for c in first] == [c.id for c in second]


def test_chunk_directory_reads_all_supported_files(settings, raw_docs_dir):
    (raw_docs_dir / "notes.md").write_text("# Heading\n\nMarkdown body text.", encoding="utf-8")
    (raw_docs_dir / "ignored.pdf").write_bytes(b"%PDF-1.4")

    chunks = TextChunker(settings=settings).chunk_directory(raw_docs_dir)

    sources = {chunk.source for chunk in chunks}
    assert "notes.md" in sources
    assert "ignored.pdf" not in sources


def test_chunk_directory_rejects_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        TextChunker().chunk_directory(tmp_path / "does-not-exist")


def test_process_documents_writes_readable_json(settings, raw_docs_dir):
    chunker = TextChunker(settings=settings)
    chunks = chunker.process_documents(raw_docs_dir, settings.processed_chunks_dir)

    written = list(settings.processed_chunks_dir.glob("*_chunks.json"))
    assert len(written) == len(set(c.source for c in chunks))

    payload = json.loads(written[0].read_text(encoding="utf-8"))
    assert payload and {"id", "text", "source", "position"} <= set(payload[0])
