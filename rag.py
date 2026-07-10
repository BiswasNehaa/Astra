from vectorstore import search
from llm import ask_ai

def answer_with_context( question : str) :
    # Step 1: find relevant chunks from our stored papers
    results= search(question, top_k=3)
    
    # Step 2: pull out just the text from the search results
    # (results['documents'] is a list containing one list of matched texts)
    chunks= results["documents"][0]
    
    # Step 3: combine the chunks into one block of text to give the AI as context
    context = "\n\n".join(chunks)

    # Step 4: build a prompt that includes both the context AND the question,
    # with clear instructions to only use the given context
    prompt = f"""Answer the question using ONLY the context below. 
If the context doesn't contain the answer, say so honestly.

Context:
{context}

Question: {question}
"""

    # Step 5: send this full prompt to the AI and return its answer
    return ask_ai(prompt)