from fastapi import FastAPI
from pydantic import BaseModel
from rag import answer_with_context

class QueryRequest(BaseModel):
    query: str
    
    
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
