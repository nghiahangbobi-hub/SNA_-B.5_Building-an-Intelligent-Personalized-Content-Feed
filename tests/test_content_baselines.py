"""PHUC-03 numerical/cache/integration tests; no network or real PLM required."""

from contextlib import nullcontext
import csv
from dataclasses import replace
from datetime import datetime
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from scripts import cache_plm_embeddings as cache_script
from scripts.run_baselines import prediction_rows, run_baselines
from src.baselines.base import ScoringContext
from src.baselines.b2_popularity import ClickIndex
from src.baselines.b3_category import CategoryBaseline
from src.baselines.b4_semantic_plm import (
    CACHE_POLICY_KEYS, CONTENT_POLICY, EmbeddingStore, SemanticBaseline,
    content_sha256, file_sha256, load_content_config,
)
from src.data_loader.mind_reader import Behavior, NewsMetadata, read_behaviors, read_news


def context(history=("A", "A", "B"), candidates=("A", "B", "C", "D", "unknown")):
    return ScoringContext("dev", "1", datetime(2019, 11, 15), candidates, history)


@pytest.fixture
def news():
    return {
        "A": NewsMetadata("sports", "football", "Football", ""),
        "B": NewsMetadata("travel", "travel", "Travel", ""),
        "C": NewsMetadata("sports", "tennis", "Tennis", ""),
        "D": NewsMetadata("sports", None, "", ""),
    }


def store_for(news):
    ids = sorted(n for n, item in news.items() if item.text)
    vectors = np.zeros((len(ids), 384), dtype=np.float32)
    for i in range(len(ids)):
        vectors[i, i % 384] = 1
    return EmbeddingStore(ids, vectors)


def synthetic_cache(path, news):
    """Explicit synthetic artifacts, never misrepresented as real PLM evidence."""
    path.mkdir()
    store = store_for(news)
    ids = list(store.indices)
    np.save(path / "embeddings.npy", store.vectors, allow_pickle=False)
    (path / "news_ids.json").write_text(json.dumps(ids), encoding="utf-8")
    manifest = {
        "format_version": 1, "status": "completed", "data_kind": "synthetic",
        "backend": "synthetic_fixture", "news_count": len(news), "encoded_news": len(ids),
        "embedding_policy": {k: CONTENT_POLICY[k] for k in CACHE_POLICY_KEYS},
        "content_sha256": content_sha256(news),
        "files": {n: file_sha256(path / n) for n in ("embeddings.npy", "news_ids.json")},
    }
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_category_hand_calculated_and_fallback(news):
    model = CategoryBaseline(news)
    # C has an unseen but VALID subcategory: do not fall back to its sports category.
    assert model.score(context()) == pytest.approx((2/3, 1/3, 0, 2/3, 0))
    assert model.diagnostics["category_fallback_candidates"] == 1
    assert model.diagnostics["missing_candidate_metadata_items"] == 1


def test_category_no_valid_subcategory_uses_category(news):
    assert CategoryBaseline(news).score(context(history=("D",))) == (1, 0, 1, 1, 0)


@pytest.mark.parametrize("history", [(), ("unknown",)])
def test_empty_unknown_history_scores_zero(news, history):
    for model in (CategoryBaseline(news), SemanticBaseline(news, store_for(news))):
        assert model.score(context(history)) == (0, 0, 0, 0, 0)


def test_semantic_mean_cosine_and_duplicates(news):
    model = SemanticBaseline(news, store_for(news))
    assert model.score(context()) == pytest.approx((2/np.sqrt(5), 1/np.sqrt(5), 0, 0, 0))
    assert model.diagnostics["empty_candidate_text_items"] == 1
    assert model.diagnostics["missing_candidate_metadata_items"] == 1
    assert model.diagnostics["usable_history_items"] == 3


def test_missing_history_excluded_from_denominator(news):
    for model in (CategoryBaseline(news), SemanticBaseline(news, store_for(news))):
        assert model.score(context(("A", "unknown"), ("A",))) == (1.0,)
        assert model.diagnostics["missing_history_metadata_items"] == 1


def test_semantic_zero_norm_and_negative_scores(news):
    vectors = store_for(news).vectors.copy()
    vectors[1] = -vectors[0]
    store = EmbeddingStore(["A", "B", "C"], vectors)
    model = SemanticBaseline(news, store)
    assert model.score(context(("A", "B"))) == (0, 0, 0, 0, 0)
    assert model.diagnostics["zero_norm_user_impressions"] == 1
    # Missing content fallback 0 can rank above negative cosine; no popularity blend.
    assert model.score(context(("A",), ("B", "D"))) == (-1, 0)


def test_history_order_invariance_and_candidate_alignment(news):
    for model in (CategoryBaseline(news), SemanticBaseline(news, store_for(news))):
        first = model.score(context(candidates=("A", "B")))
        assert model.score(context(("B", "A", "A"), ("B", "A"))) == first[::-1]


def test_current_impression_history_only_no_state_or_label_leakage(news):
    rows = [Behavior("dev", "1", "U", datetime(2019, 11, 15), 1, ("A", "B"), (1, 0),
                     "test", ("A",))]
    index = ClickIndex([], source_end=datetime(2019, 11, 15))
    for model in (CategoryBaseline(news), SemanticBaseline(news, store_for(news))):
        original = list(prediction_rows(rows, model, index, {}))
        # Scoring a later impression for the same user cannot contaminate an earlier one.
        model.score(ScoringContext("dev", "future", datetime(2019, 11, 15, 23), ("A", "B"), ("B",)))
        changed = list(prediction_rows([replace(rows[0], labels=(0, 1))], model, index, {}))
        assert [c.score for c in original[0].candidates] == [c.score for c in changed[0].candidates] == [1, 0]
        assert all(c.prior_click_count == 0 for c in changed[0].candidates)


def test_reader_preserves_history_and_text(tmp_path):
    behavior = tmp_path / "behavior.tsv"
    behavior.write_text("1\tU\t11/14/2019 12:00:00 AM\tA A B\tA-1 B-0\n", encoding="utf-8")
    row = next(read_behaviors(behavior, source="train"))
    assert row.history_ids == row.context().history_ids == ("A", "A", "B")
    assert row.history_length == 3
    assert not hasattr(row.context(), "labels")
    path = tmp_path / "news.tsv"
    path.write_text("A\tsports\tfootball\t Title \t Abstract \turl\t[]\t[]\n", encoding="utf-8")
    assert read_news([path, path])["A"].text == "Title. Abstract"


@pytest.mark.parametrize("title,abstract,expected", [(" T ", "", "T"), ("", " A ", "A"), (" ", " ", "")])
def test_text_missing_fields(title, abstract, expected):
    assert NewsMetadata(None, None, title, abstract).text == expected


def test_metadata_conflict_not_silently_overwritten(tmp_path):
    a, b = tmp_path / "a.tsv", tmp_path / "b.tsv"
    a.write_text("N\tc\ts\ttitle\ta\turl\t[]\t[]\n", encoding="utf-8")
    b.write_text("N\tc\ts\tCHANGED\ta\turl\t[]\t[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="conflicting content"):
        read_news([a, b])


@pytest.mark.parametrize("change", ["shape", "dtype", "nan", "zero", "duplicate"])
def test_invalid_embedding_matrix_rejected(change):
    ids, vectors = ["A", "B"], np.eye(2, 384, dtype=np.float32)
    if change == "shape": vectors = vectors[:, :2]
    if change == "dtype": vectors = vectors.astype(np.float64)
    if change == "nan": vectors[0, 0] = np.nan
    if change == "zero": vectors[0] = 0
    if change == "duplicate": ids = ["A", "A"]
    with pytest.raises(ValueError):
        EmbeddingStore(ids, vectors)


def test_embedding_matrix_copied_and_immutable():
    vectors = np.eye(1, 384, dtype=np.float32)
    store = EmbeddingStore(["A"], vectors)
    vectors[0, 0] = 0
    assert store.vectors[0, 0] == 1
    with pytest.raises(ValueError):
        store.vectors[0, 0] = 0


def test_cache_roundtrip_and_text_coverage(tmp_path, news):
    cache = tmp_path / "cache"
    synthetic_cache(cache, news)
    store, manifest = EmbeddingStore.load(cache, news, CONTENT_POLICY, data_kind="synthetic")
    assert manifest["backend"] == "synthetic_fixture"
    assert SemanticBaseline(news, store).score(context()) == pytest.approx((2/np.sqrt(5), 1/np.sqrt(5), 0, 0, 0))
    with pytest.raises(ValueError, match="mapping"):
        SemanticBaseline({**news, "new": NewsMetadata(None, None, "text", "")}, store)


@pytest.mark.parametrize("change", ["revision", "preprocessing", "status", "content", "mapping", "vectors", "real_data", "backend"])
def test_cache_drift_and_corruption_rejected(tmp_path, news, change):
    cache = tmp_path / "cache"
    manifest = synthetic_cache(cache, news)
    if change == "revision": manifest["embedding_policy"]["model_revision"] = "wrong"
    if change == "preprocessing": manifest["embedding_policy"]["max_seq_length"] = 100
    if change == "status": manifest["status"] = "failed"
    if change == "content": news = {**news, "A": replace(news["A"], title="changed")}
    if change == "mapping": (cache / "news_ids.json").write_text('["wrong"]', encoding="utf-8")
    if change == "vectors": np.save(cache / "embeddings.npy", np.zeros((3, 384), dtype=np.float32))
    if change == "backend": manifest["backend"] = "unknown"
    (cache / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        EmbeddingStore.load(cache, news, CONTENT_POLICY, data_kind="MIND-small" if change == "real_data" else "synthetic")


@pytest.mark.parametrize("key,value", [("seed", 1), ("normalize_news", 1), ("model_revision", "main"), ("history", "all_users")])
def test_content_config_rejects_unapproved_policy(tmp_path, key, value):
    config = load_content_config("configs/content_baselines_config.yaml")
    config[key] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="policy"):
        load_content_config(path)


@pytest.fixture
def small_inputs(tmp_path):
    train, dev, news_path = [tmp_path / name for name in ("train.tsv", "dev.tsv", "news.tsv")]
    train.write_text("1\tU\t11/13/2019 12:00:00 AM\tA B\tA-1 B-0\n"
                     "2\tU\t11/14/2019 12:00:00 AM\tA\tA-1 B-0\n", encoding="utf-8")
    dev.write_text("1\tU\t11/15/2019 12:00:00 AM\tB\tA-0 B-1\n"
                   "2\tnew\t11/15/2019 01:00:00 AM\t\tA-0 B-1\n", encoding="utf-8")
    news_path.write_text("A\tsports\tfootball\tFootball\t\turl\t[]\t[]\n"
                         "B\ttravel\ttravel\tTravel\t\turl\t[]\t[]\n", encoding="utf-8")
    return train, dev, news_path


def test_content_end_to_end_synthetic(small_inputs, tmp_path):
    train, dev, news_path = small_inputs
    cache = tmp_path / "cache"
    synthetic_cache(cache, read_news([news_path]))
    kwargs = dict(train_path=train, dev_path=dev, news_paths=[news_path],
                  output_dir=tmp_path / "tables", predictions_dir=tmp_path / "predictions",
                  content_config_path="configs/content_baselines_config.yaml",
                  embedding_cache=cache, data_kind="synthetic")
    manifest = run_baselines(**kwargs)
    assert manifest["status"] == "completed" and len(manifest["runs"]) == 4
    assert all(r["history_thresholds"]["n_train"] == 1 for r in manifest["runs"])
    assert all(r["content_diagnostics"]["empty_history_impressions"] == 1 for r in manifest["runs"] if r["split"] == "test")
    with (tmp_path / "predictions/test/B4_semantic_plm/per_impression.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert json.loads(rows[0]["scores"]) == [0, 1]
    assert json.loads(rows[1]["scores"]) == [0, 0]
    with pytest.raises(FileExistsError):
        run_baselines(**kwargs)


def test_b3_only_requires_no_cache(small_inputs, tmp_path):
    train, dev, news_path = small_inputs
    manifest = run_baselines(train_path=train, dev_path=dev, news_paths=[news_path],
                             output_dir=tmp_path / "tables", predictions_dir=tmp_path / "predictions",
                             content_config_path="configs/content_baselines_config.yaml",
                             models=["B3_category"], data_kind="synthetic")
    assert len(manifest["runs"]) == 2 and manifest["embedding_cache"] is None


@pytest.mark.parametrize("models", [[], ["B1_random"], ["B3_category", "B3_category"]])
def test_invalid_model_selection(small_inputs, tmp_path, models):
    train, dev, news_path = small_inputs
    with pytest.raises(ValueError, match="subset"):
        run_baselines(train_path=train, dev_path=dev, news_paths=[news_path], models=models,
                      output_dir=tmp_path / "tables", predictions_dir=tmp_path / "predictions",
                      content_config_path="configs/content_baselines_config.yaml")
    assert not (tmp_path / "tables").exists()


def test_missing_cache_fails_before_outputs(small_inputs, tmp_path):
    train, dev, news_path = small_inputs
    with pytest.raises(ValueError, match="requires a completed"):
        run_baselines(train_path=train, dev_path=dev, news_paths=[news_path],
                      output_dir=tmp_path / "tables", predictions_dir=tmp_path / "predictions",
                      content_config_path="configs/content_baselines_config.yaml")
    assert not (tmp_path / "tables").exists()


@pytest.mark.parametrize("fail", [False, True])
def test_cache_writer_contract_with_mock_encoder(small_inputs, tmp_path, monkeypatch, fail):
    """Mock checks orchestration ONLY, not model downloads or real encoder correctness."""
    _, _, news_path = small_inputs
    calls = {}

    class FakeParameter:
        def requires_grad_(self, flag): calls["requires_grad"] = flag

    class FakeEncoder:
        def __init__(self, model, **kwargs): calls.update(model=model, kwargs=kwargs)
        def eval(self): calls["eval"] = True
        def parameters(self): return [FakeParameter()]
        def get_sentence_embedding_dimension(self): return 384
        def encode(self, texts, **kwargs):
            if fail: raise RuntimeError("mock encode failure")
            calls.update(texts=texts, encode_kwargs=kwargs, max_length=self.max_seq_length)
            return np.eye(len(texts), 384, dtype=np.float32)

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        manual_seed=lambda seed: calls.update(seed=seed),
        use_deterministic_algorithms=lambda flag: calls.update(deterministic=flag), inference_mode=nullcontext))
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=FakeEncoder))
    monkeypatch.setattr(cache_script, "version", lambda name: "synthetic_mock")
    output = tmp_path / "cache"
    if fail:
        with pytest.raises(RuntimeError, match="mock encode failure"):
            cache_script.cache_embeddings(news_paths=[news_path], output_dir=output, data_kind="synthetic")
        assert json.loads((output / "manifest.json").read_text())["status"] == "failed"
    else:
        manifest = cache_script.cache_embeddings(news_paths=[news_path], output_dir=output,
                                                 data_kind="synthetic", local_files_only=True)
        assert manifest["status"] == "completed"
        assert calls["kwargs"]["revision"] == CONTENT_POLICY["model_revision"]
        assert calls["kwargs"]["trust_remote_code"] is False
        assert calls["kwargs"]["local_files_only"] is True
        assert calls["kwargs"]["model_kwargs"] == {"use_safetensors": True}
        assert calls["requires_grad"] is False and calls["eval"] and calls["deterministic"]
        assert calls["max_length"] == 256 and calls["texts"] == ["Football", "Travel"]
        assert calls["encode_kwargs"]["normalize_embeddings"] is True
        EmbeddingStore.load(output, read_news([news_path]), CONTENT_POLICY, data_kind="synthetic")
    with pytest.raises(FileExistsError):
        cache_script.cache_embeddings(news_paths=[news_path], output_dir=output, data_kind="synthetic")
