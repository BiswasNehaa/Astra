from langgraph.graph import StateGraph, END
from typing import TypedDict


class GraphState(TypedDict):
    question: str
    context_chunks: list
    context_chunk_metadatas: list
    answer: str
    is_supported: bool
    loop_count: int
    sources: list
    citations: list


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
    # Kept per-chunk (not deduplicated) so cite_node can map a chunk index
    # back to the paper it came from.
    state["context_chunk_metadatas"] = metadatas

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


def cite_node(state: GraphState) -> GraphState:
    from llm import ask_ai
    import json

    # Citing a "not reliably supported" fallback answer doesn't make sense -
    # only attribute claims once verification has actually passed.
    if not state.get("is_supported"):
        state["citations"] = []
        return state

    numbered_context = "\n\n".join(
        f"[{i}] {chunk}" for i, chunk in enumerate(state["context_chunks"])
    )
    cite_prompt = f"""Match each distinct claim in the ANSWER to the CONTEXT
chunk(s) (labeled [N]) that directly support it.
Respond with ONLY a JSON array, no other text, in this exact shape:
[{{"claim": "...", "supporting_chunk_ids": [0, 2]}}]
If a claim isn't directly supported by any chunk, use an empty list.

Context:
{numbered_context}

Answer:
{state["answer"]}
"""
    raw = ask_ai(cite_prompt, temperature=0)

    try:
        # Models sometimes wrap the array in prose or a code fence -
        # pull out just the [...] portion before parsing.
        start = raw.index("[")
        end = raw.rindex("]") + 1
        claims = json.loads(raw[start:end])
    except (ValueError, json.JSONDecodeError):
        claims = []

    metadatas = state.get("context_chunk_metadatas", [])
    citations = []
    for claim in claims:
        chunk_ids = claim.get("supporting_chunk_ids", [])
        matched_sources = [
            {
                "title": metadatas[i].get("title", "Unknown"),
                "url": metadatas[i].get("url", ""),
            }
            for i in chunk_ids
            if isinstance(i, int) and 0 <= i < len(metadatas)
        ]
        citations.append({"claim": claim.get("claim", ""), "sources": matched_sources})

    state["citations"] = citations
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
graph.add_node("cite", cite_node)

graph.set_entry_point("generate")
graph.add_edge("generate", "verify")

graph.add_conditional_edges(
    "verify",
    decide_next_step,
    {
        "end": "cite",
        "retry": "generate",
    },
)
graph.add_edge("cite", END)

compiled_graph = graph.compile()