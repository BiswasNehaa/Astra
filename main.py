from fastapi import FastAPI
from pydantic import BaseModel, Field
from graph import compiled_graph
from ingestion import ingest_papers
from ingestion import summarize_topic

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)


class IngestRequest(BaseModel):
    topic: str = Field(..., min_length=1)
    max_results: int = Field(default=5, ge=1, le=20)

class SummarizeRequest(BaseModel):
    topic: str = Field(..., min_length=1)
    max_results: int = Field(default=5, ge=1, le=20)
     
app= FastAPI()

@app.get("/")
def home():
    return {"message": "Astra shipping"}


@app.post("/ask")
def ask(request: QueryRequest):
    # Now using the self-correcting graph instead of a single-pass answer.
    # It generates an answer, verifies it against sources, and retries if unsupported.
    result = compiled_graph.invoke({"question": request.query})
    return {
        "answer": result["answer"],
        "supported": result["is_supported"],
        "attempts": result["loop_count"],
        "sources": result["sources"],
        "citations": result["citations"],
    }


@app.post("/ingest")
def ingest(request: IngestRequest):
    count=ingest_papers(request.topic, request.max_results)
    return {"chunks_saved": count}

@app.post("/summarize_topic")
def summarize(request: SummarizeRequest):
    return summarize_topic(request.topic, request.max_results)