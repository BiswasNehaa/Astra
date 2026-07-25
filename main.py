from fastapi import FastAPI
from pydantic import BaseModel
from graph import compiled_graph
from ingestion import ingest_papers

class QueryRequest(BaseModel):
    query: str
    
    
class IngestRequest(BaseModel):
    topic: str
    max_results: int=5
    
     
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
    }


@app.post("/ingest")
def ingest(request: IngestRequest):
    count=ingest_papers(request.topic, request.max_results)
    return {"chunks_saved": count}