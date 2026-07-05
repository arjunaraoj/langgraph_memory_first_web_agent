from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

import gradio as gr

from agent import DEMO_QUERIES, MemoryFirstLangGraphAgent, AgentState, load_settings
from ingest import ingest


_agent: MemoryFirstLangGraphAgent | None = None


def get_agent() -> MemoryFirstLangGraphAgent:
    global _agent
    if _agent is None:
        _agent = MemoryFirstLangGraphAgent()
    return _agent


def apply_runtime_options(
    agent: MemoryFirstLangGraphAgent,
    threshold: float,
    top_k_memory: int,
    top_k_web: int,
    force_web: bool,
) -> None:
    """Override selected settings for this UI session without editing .env."""
    agent.settings.memory_similarity_threshold = 2.0 if force_web else float(threshold)
    agent.settings.top_k_memory = int(top_k_memory)
    agent.settings.top_k_web = int(top_k_web)
    agent.web.settings.top_k_web = int(top_k_web)


def format_metadata(result: AgentState) -> str:
    metadata = {
        "memory_hit": result.get("memory_hit", False),
        "web_search_used": result.get("web_search_used", False),
        "redis_similarity_score": result.get("top_similarity", 0.0),
        "force_web": result.get("force_web", False),
        "urls": result.get("urls", []),
        "chunks_stored": result.get("chunks_stored", 0),
        "analytics": result.get("analytics", {}),
        "warnings": result.get("warnings", []),
    }
    return json.dumps(metadata, indent=2, ensure_ascii=False)


def run_agent(
    question: str,
    threshold: float,
    top_k_memory: int,
    top_k_web: int,
    force_web: bool,
    show_raw_state: bool,
) -> Tuple[str, str]:
    question = (question or "").strip()
    if not question:
        return "Please enter a question.", "{}"

    try:
        agent = get_agent()
        apply_runtime_options(agent, threshold, top_k_memory, top_k_web, force_web)
        result = agent.run(question)
        answer = result.get("answer", "No answer generated.")
        metadata = format_metadata(result)

        if show_raw_state:
            safe_state = dict(result)
            safe_state.pop("query_vector", None)
            metadata = json.dumps(safe_state, indent=2, default=str, ensure_ascii=False)

        return answer, metadata
    except Exception as exc:
        error = (
            "The agent failed while processing the question.\n\n"
            f"Error: {exc}\n\n"
            "Check OPENAI_API_KEY, WEB_SEARCH_API_KEY, Redis status, and .env configuration."
        )
        return error, json.dumps({"error": str(exc)}, indent=2)


def chat_fn(
    message: str,
    history: List[Dict[str, str]],
    threshold: float,
    top_k_memory: int,
    top_k_web: int,
    force_web: bool,
) -> str:
    answer, metadata = run_agent(
        question=message,
        threshold=threshold,
        top_k_memory=top_k_memory,
        top_k_web=top_k_web,
        force_web=force_web,
        show_raw_state=False,
    )
    return f"{answer}\n\n---\n\n### Run metadata\n```json\n{metadata}\n```"


def run_selected_demo(
    selected_question: str,
    threshold: float,
    top_k_memory: int,
    top_k_web: int,
    force_web: bool,
    show_raw_state: bool,
) -> Tuple[str, str]:
    return run_agent(
        selected_question,
        threshold,
        top_k_memory,
        top_k_web,
        force_web,
        show_raw_state,
    )


def run_all_demos(
    threshold: float,
    top_k_memory: int,
    top_k_web: int,
    force_web: bool,
) -> Tuple[str, str]:
    answers = []
    metadata_rows = []
    for idx, question in enumerate(DEMO_QUERIES, start=1):
        answer, metadata = run_agent(
            question,
            threshold,
            top_k_memory,
            top_k_web,
            force_web,
            show_raw_state=False,
        )
        answers.append(f"## Demo {idx}: {question}\n\n{answer}")
        metadata_rows.append(
            {
                "demo": idx,
                "question": question,
                "metadata": json.loads(metadata),
            }
        )
    return "\n\n---\n\n".join(answers), json.dumps(metadata_rows, indent=2, ensure_ascii=False)


def ingest_sample_data(reset: bool) -> str:
    try:
        global _agent
        count = ingest(reset=reset)
        _agent = None  # reload agent after reset/ingest
        return f"Ingested {count} chunks into Redis memory. reset={reset}"
    except Exception as exc:
        return f"Ingestion failed: {exc}"


def read_analytics() -> str:
    settings = load_settings()
    path: Path = settings.log_path
    if not path.exists():
        return "No logs found yet. Run the agent first."

    rows: List[Dict[str, Any]] = []
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

    report = {
        "total_turns": len(rows),
        "count_by_topic": dict(topics),
        "count_by_question_type": dict(qtypes),
        "count_by_memory_hit_vs_miss": dict(hit_miss),
        "most_common_user_intents": dict(intents.most_common(10)),
        "most_frequently_used_sources": dict(sources.most_common(10)),
    }
    return json.dumps(report, indent=2, ensure_ascii=False)


def clear_logs() -> str:
    settings = load_settings()
    path: Path = settings.log_path
    if path.exists():
        path.unlink()
        return f"Deleted log file: {path}"
    return "No log file found."


def current_config() -> str:
    s = load_settings()
    data = {
        "redis_url": s.redis_url,
        "memory_similarity_threshold": s.memory_similarity_threshold,
        "conversation_model": s.conversation_model,
        "analytics_model": s.analytics_model,
        "embedding_model": s.embedding_model,
        "embedding_dimension": s.embedding_dimension,
        "redis_index_name": s.redis_index_name,
        "redis_key_prefix": s.redis_key_prefix,
        "top_k_memory": s.top_k_memory,
        "top_k_web": s.top_k_web,
        "request_timeout_seconds": s.request_timeout_seconds,
        "chunk_size_chars": s.chunk_size_chars,
        "chunk_overlap_chars": s.chunk_overlap_chars,
        "log_path": str(s.log_path),
        "openai_api_key_loaded": bool(s.openai_api_key),
        "web_search_api_key_loaded": bool(s.web_search_api_key),
    }
    return json.dumps(data, indent=2, ensure_ascii=False)


with gr.Blocks(title="LangGraph Memory-First Web Agent") as demo:
    gr.Markdown(
        """
        # LangGraph Memory-First Web Agent

        Ask a question. The agent checks Redis vector memory first. If similarity is below the threshold, it searches the web, fetches pages, converts content to Markdown, stores chunks into Redis, and answers with URLs.
        """
    )

    with gr.Accordion("Runtime options", open=True):
        with gr.Row():
            threshold = gr.Slider(
                minimum=0.1,
                maximum=0.95,
                value=load_settings().memory_similarity_threshold,
                step=0.01,
                label="Memory similarity threshold",
                info="Higher = stricter memory hits. Default assignment value is 0.7.",
            )
            top_k_memory = gr.Slider(
                minimum=1,
                maximum=10,
                value=load_settings().top_k_memory,
                step=1,
                label="Top K Redis memory chunks",
            )
            top_k_web = gr.Slider(
                minimum=1,
                maximum=5,
                value=load_settings().top_k_web,
                step=1,
                label="Top K web results",
            )

        with gr.Row():
            force_web = gr.Checkbox(
                label="Force web search",
                value=False,
                info="Useful for latest/current questions or testing web fallback.",
            )
            show_raw_state = gr.Checkbox(
                label="Show raw graph state",
                value=False,
            )

    with gr.Tabs():
        with gr.Tab("Ask question"):
            question = gr.Textbox(
                label="User question",
                placeholder="Example: What is the Singapore Tourism Board and what are its main responsibilities?",
                lines=3,
            )
            ask_btn = gr.Button("Run agent", variant="primary")
            answer_output = gr.Markdown(label="Answer")
            metadata_output = gr.Code(label="Metadata", language="json")

            ask_btn.click(
                run_agent,
                inputs=[
                    question,
                    threshold,
                    top_k_memory,
                    top_k_web,
                    force_web,
                    show_raw_state,
                ],
                outputs=[answer_output, metadata_output],
            )

        with gr.Tab("Chat mode"):
            gr.ChatInterface(
                fn=chat_fn,
                additional_inputs=[
                    threshold,
                    top_k_memory,
                    top_k_web,
                    force_web,
                ],
                title="Chat Mode",
                description="Each message runs the LangGraph memory-first workflow.",
            )

        with gr.Tab("Demo scenarios"):
            selected_demo = gr.Dropdown(
                choices=DEMO_QUERIES,
                value=DEMO_QUERIES[0],
                label="Demo question",
            )

            with gr.Row():
                run_demo_btn = gr.Button("Run selected demo", variant="primary")
                run_all_btn = gr.Button("Run all demo questions")

            demo_answer = gr.Markdown(label="Demo answer")
            demo_metadata = gr.Code(label="Demo metadata", language="json")

            run_demo_btn.click(
                run_selected_demo,
                inputs=[
                    selected_demo,
                    threshold,
                    top_k_memory,
                    top_k_web,
                    force_web,
                    show_raw_state,
                ],
                outputs=[demo_answer, demo_metadata],
            )

            run_all_btn.click(
                run_all_demos,
                inputs=[
                    threshold,
                    top_k_memory,
                    top_k_web,
                    force_web,
                ],
                outputs=[demo_answer, demo_metadata],
            )

        with gr.Tab("Data and analytics"):
            reset_checkbox = gr.Checkbox(
                label="Reset Redis index before ingest",
                value=True,
            )
            ingest_btn = gr.Button("Ingest sample data", variant="primary")
            ingest_status = gr.Textbox(label="Ingestion status")

            ingest_btn.click(
                ingest_sample_data,
                inputs=[reset_checkbox],
                outputs=[ingest_status],
            )

            with gr.Row():
                analytics_btn = gr.Button("Show analytics")
                clear_logs_btn = gr.Button("Clear logs")

            analytics_output = gr.Code(label="Analytics", language="json")
            clear_logs_output = gr.Textbox(label="Log status")

            analytics_btn.click(
                read_analytics,
                outputs=[analytics_output],
            )

            clear_logs_btn.click(
                clear_logs,
                outputs=[clear_logs_output],
            )

        with gr.Tab("Configuration"):
            config_btn = gr.Button("Show loaded configuration")
            config_output = gr.Code(label="Configuration", language="json")

            config_btn.click(
                current_config,
                outputs=[config_output],
            )


if __name__ == "__main__":
    demo.launch()
