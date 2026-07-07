from fastapi import FastAPI
from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str
    
    
app= FastAPI()

@app.get("/")
def home():
    return {"message": "Astra shipping"}

@app.post("/ask")
def ask(request: QueryRequest):
    return {"you_asked": request.query}