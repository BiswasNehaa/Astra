from fastapi import FastAPI
from pydantic import BaseModel
from llm import ask_ai

class QueryRequest(BaseModel):
    query: str
    
    
app= FastAPI()

@app.get("/")
def home():
    return {"message": "Astra shipping"}

@app.post("/ask")
def ask(request: QueryRequest):
    answer= ask_ai(request.query)
    return {"answer": answer}