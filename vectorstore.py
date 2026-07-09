import chromadb
from embeddings import get_embedding

# PersistentClient saves data to a folder on disk, so it survives
# even after you close Python or restart your computer.
# (The alternative, chromadb.Client(), only keeps data in memory - gone on exit.)
client = chromadb.PersistentClient(path="./data/chroma")

# A "collection" is like a table in a regular database -
# a named group of related vectors. We'll store all our paper chunks here.
collection = client.get_or_create_collection(name="papers")


def add_chunk(chunk_id: str, text:str, metadata: dict):
    # Convert the text into an embedding
    embedding= get_embedding(text)
    
    # Save into Chroma - it needs 4 things:
    # - ids: a unique identifier for this piece of text (so we can find/update it later)
    # - embeddings: the numbers representing meaning, used for searching
    # - documents: the original text itself, so we can show it as an answer later
    # - metadatas: extra info about this chunk (e.g. paper title, year, source)
    collection.add(
        ids=[chunk_id],
        embeddings=[embedding],
        documents=[text],
        metadatas=[metadata],
    )
    
def search(query: str, top_k: int=3):
    # Convert the user's question into an embedding, same way we did for stored chunks.
    # This lets us compare "meaning" between the question and stored text.
    query_embedding= get_embedding(query)
    
    # Ask Chroma to find the closest matches by embedding distance.
    # top_k = how many results to return (e.g. 3 = the 3 most relevant chunks).
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    return results