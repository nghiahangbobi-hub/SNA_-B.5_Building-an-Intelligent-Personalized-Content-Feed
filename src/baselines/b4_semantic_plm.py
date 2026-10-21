"""Frozen content-only embeddings and mean-history cosine scoring (no PLM import)."""

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from src.baselines.base import BaseRecommender, ScoringContext
from src.data_loader.mind_reader import NewsMetadata


CONTENT_POLICY = {
    "seed": 0, "models": ["B3_category", "B4_semantic_plm"],
    "history": "current_impression_all_ids",
    "category_score": "subcategory_probability_category_fallback",
    "empty_profile_score": 0.0,
    "model_id": "sentence-transformers/all-MiniLM-L6-v2",
    "model_revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    "text": "title_dot_abstract_strip_outer", "max_seq_length": 256,
    "normalize_news": True, "user_pooling": "mean_then_cosine", "dimension": 384,
    "device": "cpu", "batch_size": 64,
}
CACHE_POLICY_KEYS = ("seed", "model_id", "model_revision", "text", "max_seq_length",
                     "normalize_news", "dimension", "device", "batch_size")


def load_content_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    # Type checks prevent YAML bool/int equality from concealing policy changes.
    if not isinstance(config, dict) or set(config) != set(CONTENT_POLICY) or any(
        type(config[k]) is not type(v) or config[k] != v for k, v in CONTENT_POLICY.items()
    ):
        raise ValueError("content config differs from approved PHUC-03 policy")
    return config


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_sha256(news: dict[str, NewsMetadata]) -> str:
    payload = [(news_id, news[news_id].text) for news_id in sorted(news)]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


class EmbeddingStore:
    """Immutable, unit-normalized matrix; only nonempty-text news have rows."""

    def __init__(self, ids: list[str], vectors: np.ndarray, *, dimension: int = 384):
        if not isinstance(ids, list) or any(not isinstance(n, str) or not n.strip() for n in ids):
            raise ValueError("cache IDs must be nonempty strings")
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate cache news IDs")
        if vectors.dtype != np.float32 or vectors.shape != (len(ids), dimension):
            raise ValueError("invalid embedding dtype/shape")
        if not np.isfinite(vectors).all():
            raise ValueError("nonfinite cache vectors")
        if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5, rtol=1e-5):
            raise ValueError("cache vectors must have unit norm (nonzero)")
        self.vectors = np.array(vectors, copy=True)
        self.vectors.flags.writeable = False
        self.indices = {news_id: i for i, news_id in enumerate(ids)}

    def validate_coverage(self, news: dict[str, NewsMetadata]) -> None:
        expected = {news_id for news_id, item in news.items() if item.text}
        if set(self.indices) != expected:
            raise ValueError("cache mapping does not match nonempty news text; rebuild cache")

    @classmethod
    def load(cls, directory: str | Path, news: dict[str, NewsMetadata], config: dict,
             *, data_kind: str) -> tuple["EmbeddingStore", dict]:
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "completed" or manifest.get("format_version") != 1:
            raise ValueError("incomplete/unsupported embedding cache")
        expected_policy = {key: config[key] for key in CACHE_POLICY_KEYS}
        if manifest.get("embedding_policy") != expected_policy:
            raise ValueError("embedding cache model/revision/preprocessing mismatch")
        if manifest.get("data_kind") != data_kind or manifest.get("content_sha256") != content_sha256(news):
            raise ValueError("embedding cache dataset/content mismatch")
        backend = manifest.get("backend")
        if backend != "sentence_transformers" and not (data_kind == "synthetic" and backend == "synthetic_fixture"):
            raise ValueError("synthetic/unknown encoder cannot serve a real-data run")
        for name in ("news_ids.json", "embeddings.npy"):
            if manifest.get("files", {}).get(name) != file_sha256(directory / name):
                raise ValueError(f"embedding cache checksum mismatch: {name}")
        ids = json.loads((directory / "news_ids.json").read_text(encoding="utf-8"))
        vectors = np.load(directory / "embeddings.npy", allow_pickle=False)
        store = cls(ids, vectors, dimension=config["dimension"])
        store.validate_coverage(news)
        if manifest.get("encoded_news") != len(ids) or manifest.get("news_count") != len(news):
            raise ValueError("embedding cache count mismatch")
        return store, manifest


class SemanticBaseline(BaseRecommender):
    name = "B4_semantic_plm"

    def __init__(self, news: dict[str, NewsMetadata], store: EmbeddingStore):
        store.validate_coverage(news)
        self.news = dict(news)
        self.store = store
        self.diagnostics: Counter = Counter()

    def score(self, context: ScoringContext) -> tuple[float, ...]:
        stats = self.diagnostics
        stats["impressions"] += 1
        stats["empty_history_impressions"] += not context.history_ids
        stats["history_items"] += len(context.history_ids)
        positions = []
        for news_id in context.history_ids:
            if news_id not in self.news:
                stats["missing_history_metadata_items"] += 1
            elif not self.news[news_id].text:
                stats["empty_history_text_items"] += 1
            else:
                positions.append(self.store.indices[news_id])
        stats["usable_history_items"] += len(positions)
        user = None
        if positions:
            # Float64 accumulation reduces cancellation/long-history rounding error.
            pooled = self.store.vectors[positions].mean(axis=0, dtype=np.float64)
            norm = np.linalg.norm(pooled)
            if norm > 1e-12:
                user = pooled / norm
            else:
                stats["zero_norm_user_impressions"] += 1
        stats["no_usable_history_impressions"] += not positions
        scores = []
        for news_id in context.candidate_ids:
            stats["candidate_items"] += 1
            missing = news_id not in self.news
            empty = not missing and not self.news[news_id].text
            stats["missing_candidate_metadata_items"] += missing
            stats["empty_candidate_text_items"] += empty
            if missing or empty or user is None:
                stats["fallback_zero_candidates"] += 1
                scores.append(0.0)
            else:
                score = float(self.store.vectors[self.store.indices[news_id]] @ user)
                scores.append(float(np.clip(score, -1.0, 1.0)))
        return tuple(scores)
