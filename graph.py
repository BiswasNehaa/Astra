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

    # Widen retrieval on each retry so a failed verification actually gets a
    # different, larger pool of context to work with, instead of repeating
    # the exact same search and hoping for a different generation by chance.
    attempt = state.get("loop_count", 0)
    top_k = 3 + (2 * attempt)

    results = search(state["question"], top_k=top_k)
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
Check whether EVERY factual claim in the ANSWER below is directly stated in
the CONTEXT. If any part of the answer is not directly supported by the
context, the whole answer counts as unsupported - do not give partial credit.
Respond with ONLY one word, exactly "yes" or "no", with no other text.

Context:
{context}

Answer:
{state["answer"]}
"""
    # temperature=0 so the same answer+context pair gets a consistent verdict
    # every time, instead of the yes/no judgment itself being non-deterministic.
    verdict = ask_ai(verify_prompt, temperature=0).strip().lower()
    state["is_supported"] = verdict.startswith("yes")

    # If we've exhausted our retries and still can't verify the answer,
    # don't silently hand back a possibly-hallucinated draft as if it were
    # a normal answer - say so explicitly.
    if not state["is_supported"] and state["loop_count"] >= 2:
        state["answer"] = (
            "I don't have enough reliably supported information in the "
            "retrieved sources to answer this confidently. Best attempt "
            f"(not fully verified): {state['answer']}"
        )

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