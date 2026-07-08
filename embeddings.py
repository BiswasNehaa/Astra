from sentence_transformers import SentenceTransformer

# Load a Pre-trau=ined embedding model .
# This downloads the model files the first time (Cached after that).
# and runs locally on your machine - no API key, no cost.
model= SentenceTransformer("BAAI/bge-small-en-v1.5")


def get_embedding(text: str):
    # model.encode() converts text into a list of numbers representing its meaning.
    # Similar-meaning text will produce numbers that are close together.
    embedding= model.encode(text)
    
    # The model returns a NumPy array by default - convert to a plain Python
    # list so it's easier to store and work with later (e.g. saving to a database).
    return model.encode(text).tolist()