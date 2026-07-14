from langgraph.graph import StateGraph, END
from typing import TypedDict


# LangGraph needs a defined "state" - basically a dictionary describing
# what information flows between each step of the pipeline.
class GraphState(TypedDict):
    question: str
    context_chunks: list
    answer: str
    is_supported: bool
    
def generate_node(state: GraphState) -> GraphState:
    from vectorstore import search
    from llm import ask_ai

    results = search(state["question"], top_k=3)
    chunks = results["documents"][0]
    state["context_chunks"] = chunks

    context = "\n\n".join(chunks)
    prompt = f"""Answer the question using ONLY the context below.
If the context doesn't contain the answer, say so honestly.

Context:
{context}

Question: {state["question"]}
"""
    state["answer"] = ask_ai(prompt)
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


graph = StateGraph(GraphState)
graph.add_node("generate", generate_node)
graph.add_node("verify", verify_node)

graph.set_entry_point("generate")
graph.add_edge("generate", "verify")
graph.add_edge("verify", END)

compiled_graph = graph.compile()