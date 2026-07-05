from __future__ import annotations

from html import escape
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph


# -------------------------------------------------------
# Project folder
# -------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "langgraph_html"
OUTPUT_FILE = OUTPUT_DIR / "memory_first_langgraph.html"


# -------------------------------------------------------
# State schema for graph visualization
# -------------------------------------------------------
class AgentState(TypedDict, total=False):
    question: str
    query_vector: list[float]
    memory_results: list
    top_similarity: float
    memory_hit: bool
    force_web: bool
    search_results: list
    web_context: list
    web_search_used: bool
    urls: list[str]
    chunks_stored: int
    answer: str
    analytics: dict
    warnings: list[str]
    blocked: bool


# -------------------------------------------------------
# Dummy node functions
# These are only for visualization.
# They do not call OpenAI, Redis, or Tavily.
# -------------------------------------------------------
def guardrails(state: AgentState) -> AgentState:
    return state


def embed_query(state: AgentState) -> AgentState:
    return state


def search_memory(state: AgentState) -> AgentState:
    return state


def answer_from_memory(state: AgentState) -> AgentState:
    return state


def web_search_fetch_store(state: AgentState) -> AgentState:
    return state


def answer_from_web(state: AgentState) -> AgentState:
    return state


def log_turn(state: AgentState) -> AgentState:
    return state


def route_guardrails(state: AgentState) -> str:
    return "ok"


def route_memory(state: AgentState) -> str:
    return "hit"


# -------------------------------------------------------
# Build the same LangGraph structure as your agent.py
# -------------------------------------------------------
def build_visual_graph():
    graph = StateGraph(AgentState)

    graph.add_node("guardrails", guardrails)
    graph.add_node("embed_query", embed_query)
    graph.add_node("search_memory", search_memory)
    graph.add_node("answer_from_memory", answer_from_memory)
    graph.add_node("web_search_fetch_store", web_search_fetch_store)
    graph.add_node("answer_from_web", answer_from_web)
    graph.add_node("log_turn", log_turn)

    graph.add_edge(START, "guardrails")

    graph.add_conditional_edges(
        "guardrails",
        route_guardrails,
        {
            "blocked": "log_turn",
            "ok": "embed_query",
        },
    )

    graph.add_edge("embed_query", "search_memory")

    graph.add_conditional_edges(
        "search_memory",
        route_memory,
        {
            "hit": "answer_from_memory",
            "miss": "web_search_fetch_store",
        },
    )

    graph.add_edge("web_search_fetch_store", "answer_from_web")
    graph.add_edge("answer_from_memory", "log_turn")
    graph.add_edge("answer_from_web", "log_turn")
    graph.add_edge("log_turn", END)

    return graph.compile()


# -------------------------------------------------------
# Export LangGraph as HTML
# -------------------------------------------------------
def export_graph_html() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    compiled_graph = build_visual_graph()

    mermaid_code = compiled_graph.get_graph().draw_mermaid()

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Memory First LangGraph Workflow</title>

    <style>
        body {{
            font-family: Arial, sans-serif;
            background: #f5f6fa;
            margin: 0;
            padding: 30px;
        }}

        .container {{
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.10);
        }}

        h1 {{
            margin-top: 0;
            color: #222;
        }}

        p {{
            color: #555;
            font-size: 15px;
        }}

        .mermaid {{
            text-align: center;
            margin-top: 30px;
        }}
    </style>

    <script type="module">
        import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs";

        mermaid.initialize({{
            startOnLoad: true,
            theme: "default",
            securityLevel: "loose",
            flowchart: {{
                curve: "linear"
            }}
        }});
    </script>
</head>

<body>
    <div class="container">
        <h1>Memory First LangGraph Workflow</h1>

        <p>
            This diagram shows the flow of the Memory-First Web Agent:
            guardrails, query embedding, Redis memory search, memory answer,
            web fallback, web answer, and turn logging.
        </p>

        <pre class="mermaid">
{escape(mermaid_code)}
        </pre>
    </div>
</body>
</html>
"""

    OUTPUT_FILE.write_text(html_content, encoding="utf-8")

    return OUTPUT_FILE


# -------------------------------------------------------
# Main runner
# -------------------------------------------------------
if __name__ == "__main__":
    html_path = export_graph_html()
    print(f"LangGraph HTML file created successfully:")
    print(html_path)