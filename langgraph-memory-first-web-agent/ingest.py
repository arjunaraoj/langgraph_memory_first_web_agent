from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent import LLMService, RedisMemory, chunk_text, load_settings

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "sample_knowledge.json"


def ingest(reset: bool = False) -> int:
    settings = load_settings()
    memory = RedisMemory(settings)
    llm = LLMService(settings)

    if reset:
        memory.reset()

    records = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    texts = []
    metas = []
    for record in records:
        base_meta = record.get("metadata", {})
        base_meta["title"] = record.get("title", "")
        for idx, chunk in enumerate(chunk_text(record["text"], settings.chunk_size_chars, settings.chunk_overlap_chars), start=1):
            meta = dict(base_meta)
            meta["chunk_index"] = idx
            texts.append(chunk)
            metas.append(meta)

    embeddings = llm.embed(texts)
    count = 0
    for text, embedding, meta in zip(texts, embeddings, metas):
        memory.add(text, embedding, meta)
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest sample knowledge into Redis vector memory")
    parser.add_argument("--reset", action="store_true", help="Drop existing index and documents first")
    args = parser.parse_args()
    count = ingest(reset=args.reset)
    print(f"Ingested {count} chunks into Redis memory.")


if __name__ == "__main__":
    main()
