from langgraph.graph import StateGraph, END
from typing import TypedDict


# LangGraph needs a defined "state" - basically a dictionary describing
# what information flows between each step of the pipeline.
class GraphState(TypedDict):
    question: str
    answer: str
    
def generate_node(state: GraphState) -> GraphState:
    # For now, just reuse your existing rag.py logic directly.
    from rag import answer_with_context
    state["answer"] = answer_with_context(state["question"])
    return state


# Build the graph
graph = StateGraph(GraphState)
graph.add_node("generate", generate_node)
graph.set_entry_point("generate")
graph.add_edge("generate", END)

compiled_graph = graph.compile()
