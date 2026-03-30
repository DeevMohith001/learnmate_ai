from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from modules.utils import clean_token, ensure_directory

try:
    import faiss
except Exception:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


_embed_model = None
_embed_model_error = None


def _get_embed_model():
    global _embed_model, _embed_model_error

    if _embed_model is not None:
        return _embed_model

    if SentenceTransformer is None:
        _embed_model_error = "sentence-transformers is not installed."
        return None

    try:
        _embed_model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1", local_files_only=True)
        _embed_model_error = None
        return _embed_model
    except Exception as exc:
        _embed_model_error = str(exc)
        return None


def _index_paths(index_path: str) -> tuple[Path, Path]:
    return Path(f"{index_path}.index"), Path(f"{index_path}_texts.json")


def _token_overlap_score(query: str, text: str) -> int:
    query_tokens = {clean_token(token) for token in query.split()}
    text_tokens = {clean_token(token) for token in text.split()}
    query_tokens.discard("")
    text_tokens.discard("")
    return len(query_tokens & text_tokens)


def build_vectorstore(texts: list[str], index_path: str = "embeddings/vectordb"):
    """Persist chunk text and optional embeddings for retrieval."""
    ensure_directory("embeddings")
    index_file, text_file = _index_paths(index_path)

    with text_file.open("w", encoding="utf-8") as file:
        json.dump(list(texts), file, ensure_ascii=False, indent=2)

    embed_model = _get_embed_model()
    if faiss is None or embed_model is None or not texts:
        return

    embeddings = np.asarray(embed_model.encode(texts), dtype="float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(index_file))


def retrieve_relevant_chunks(
    query: str,
    k: int = 3,
    index_path: str = "embeddings/vectordb",
    score_threshold: float = 2.0,
):
    """Return the best matching chunks for the supplied query."""
    index_file, text_file = _index_paths(index_path)
    if not text_file.exists():
        return []

    with text_file.open("r", encoding="utf-8") as file:
        texts = json.load(file)

    embed_model = _get_embed_model()
    if faiss is not None and embed_model is not None and index_file.exists():
        query_embedding = np.asarray(embed_model.encode([query]), dtype="float32")
        index = faiss.read_index(str(index_file))
        distances, indices = index.search(query_embedding, min(k, len(texts)))

        relevant_chunks = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx >= 0 and distance <= score_threshold:
                relevant_chunks.append(texts[idx])
        if relevant_chunks:
            return relevant_chunks

    ranked_texts = sorted(texts, key=lambda text: _token_overlap_score(query, text), reverse=True)
    return [text for text in ranked_texts[:k] if _token_overlap_score(query, text) > 0]
