from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

from agent import load_settings


def main() -> None:
    settings = load_settings()
    path: Path = settings.log_path
    if not path.exists():
        print("No logs found yet. Run python agent.py first.")
        return

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    topics = Counter((r.get("analytics") or {}).get("topic", "unknown") for r in rows)
    qtypes = Counter((r.get("analytics") or {}).get("question_type", "unknown") for r in rows)
    intents = Counter((r.get("analytics") or {}).get("intent", "unknown") for r in rows)
    hit_miss = Counter("memory_hit" if r.get("memory_hit") else "memory_miss" for r in rows)
    sources = Counter()
    for r in rows:
        for url in r.get("urls_fetched", []):
            host = urlparse(url).netloc
            if host:
                sources[host] += 1

    print("Analytics report")
    print(f"total_turns: {len(rows)}")
    print(f"count_by_topic: {dict(topics)}")
    print(f"count_by_question_type: {dict(qtypes)}")
    print(f"count_by_memory_hit_vs_miss: {dict(hit_miss)}")
    print(f"most_common_user_intents: {dict(intents.most_common(10))}")
    print(f"most_frequently_used_sources: {dict(sources.most_common(10))}")


if __name__ == "__main__":
    main()
