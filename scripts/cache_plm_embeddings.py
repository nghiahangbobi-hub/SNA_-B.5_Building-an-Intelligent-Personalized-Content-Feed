"""Create a new, provenance-checked frozen MiniLM cache; never overwrite one."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import random

import numpy as np

from src.baselines.b4_semantic_plm import (
    CACHE_POLICY_KEYS, EmbeddingStore, content_sha256, file_sha256, load_content_config,
)
from src.data_loader.mind_reader import read_news


def cache_embeddings(*, news_paths: list[str | Path], output_dir: str | Path,
                     config_path: str | Path = "configs/content_baselines_config.yaml",
                     data_kind: str = "MIND-small", local_files_only: bool = False) -> dict:
    config = load_content_config(config_path)
    directory = Path(output_dir)
    if directory.exists():
        raise FileExistsError("use a new embedding cache directory; existing evidence is preserved")
    news = read_news(news_paths)
    ids = sorted(n for n, item in news.items() if item.text)
    if not ids:
        raise ValueError("no nonempty news text to encode")
    # Lazy imports keep B1-B3 and numerical B4 tests usable without a PLM runtime.
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("PLM dependencies missing; install requirements.txt in .venv") from exc
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    manifest = {
        "format_version": 1, "status": "running", "data_kind": data_kind,
        "backend": "sentence_transformers", "python": platform.python_version(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "embedding_policy": {k: config[k] for k in CACHE_POLICY_KEYS},
        "content_sha256": content_sha256(news), "news_count": len(news),
        "encoded_news": len(ids), "empty_text_news": len(news) - len(ids),
        "libraries": {name: version(name) for name in ("numpy", "torch", "sentence-transformers", "transformers")},
        "inputs": [{"path": str(p), "sha256": file_sha256(Path(p))} for p in news_paths],
        "code_sha256": {name: file_sha256(Path(__file__).resolve().parents[1] / name) for name in
                        ("scripts/cache_plm_embeddings.py", "src/baselines/b4_semantic_plm.py",
                         "src/data_loader/mind_reader.py", "requirements.txt")},
        "files": {},
    }
    directory.mkdir(parents=True, exist_ok=False)

    def save_manifest():
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n",
                                               encoding="utf-8")

    save_manifest()
    try:
        model = SentenceTransformer(config["model_id"], revision=config["model_revision"],
                                    device=config["device"], trust_remote_code=False,
                                    local_files_only=local_files_only,
                                    model_kwargs={"use_safetensors": True})
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.max_seq_length = config["max_seq_length"]
        if model.get_sentence_embedding_dimension() != config["dimension"]:
            raise ValueError("unexpected PLM dimension")
        print(f"Encoding {len(ids)} news on {config['device']} (frozen PLM)", flush=True)
        with torch.inference_mode():
            vectors = model.encode([news[n].text for n in ids], batch_size=config["batch_size"],
                                   show_progress_bar=True, convert_to_numpy=True,
                                   normalize_embeddings=True)
        vectors = np.asarray(vectors, dtype=np.float32)
        EmbeddingStore(ids, vectors, dimension=config["dimension"]).validate_coverage(news)
        np.save(directory / "embeddings.npy", vectors, allow_pickle=False)
        (directory / "news_ids.json").write_text(json.dumps(ids) + "\n", encoding="utf-8")
        manifest["files"] = {name: file_sha256(directory / name) for name in ("news_ids.json", "embeddings.npy")}
        manifest["status"] = "completed"
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save_manifest()
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        save_manifest()
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--news", nargs="+", default=["data/raw/train/news.tsv", "data/raw/dev/news.tsv"])
    parser.add_argument("--config", default="configs/content_baselines_config.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-kind", default="MIND-small")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    cache_embeddings(news_paths=args.news, output_dir=args.output_dir, config_path=args.config,
                     data_kind=args.data_kind, local_files_only=args.local_files_only)


if __name__ == "__main__":
    main()
