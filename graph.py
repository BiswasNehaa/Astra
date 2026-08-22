from langgraph.graph import StateGraph, END
from typing import TypedDict


class GraphState(TypedDict):
    question: str
    context_chunks: list
    answer: str
    is_supported: bool
    loop_count: int
    sources: list


def generate_node(state: GraphState) -> GraphState:
    from vectorstore import search
    from llm import ask_ai

    results = search(state["question"], top_k=3)
    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    state["context_chunks"] = chunks

    # Build a simple sources list - one entry per chunk, deduplicated by title
    sources = []
    seen_titles = set()
    for m in metadatas:
        title = m.get("title", "Unknown")
        if title not in seen_titles:
            sources.append({"title": title, "url": m.get("url", "")})
            seen_titles.add(title)
    state["sources"] = sources

    context = "\n\n".join(chunks)
    prompt = f"""Answer the question using ONLY the context below.
If the context doesn't contain the answer, say so honestly.

Context:
{context}

Question: {state["question"]}
"""
    state["answer"] = ask_ai(prompt)
    state["loop_count"] = state.get("loop_count", 0) + 1
    return state


def verify_node(state: GraphState) -> GraphState:
    from llm import ask_ai

    context = "\n\n".join(state["context_chunks"])
    verify_prompt = f"""You are a strict fact-checker.
Check if the ANSWER below is fully supported by the CONTEXT.
Respond with ONLY one word: "yes" or "no".

Context:
{context}

Answer:
{state["answer"]}
"""
    verdict = ask_ai(verify_prompt).strip().lower()
    state["is_supported"] = "yes" in verdict
    return state


def decide_next_step(state: GraphState) -> str:
    if state["is_supported"]:
        return "end"
    if state["loop_count"] >= 2:
        return "end"
    return "retry"


graph = StateGraph(GraphState)
graph.add_node("generate", generate_node)
graph.add_node("verify", verify_node)

graph.set_entry_point("generate")
graph.add_edge("generate", "verify")

graph.add_conditional_edges(
    "verify",
    decide_next_step,
    {
        "end": END,
        "retry": "generate",
    },
)

compiled_graph = graph.compile()