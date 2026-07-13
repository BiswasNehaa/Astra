from fastapi import FastAPI
from pydantic import BaseModel
from rag import answer_with_context
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
     # Now using RAG - the answer is grounded in stored context,
    # not just the AI's raw training knowledge.
    answer = answer_with_context(request.query)
    return {"answer": answer}


@app.post("/ingest")
def ingest(request: IngestRequest):
    count=ingest_papers(request.topic, request.max_results)
    return {"chunks_saved": count}