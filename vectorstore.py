import chromadb

# PersistentClient saves data to a folder on disk, so it survives
# even after you close Python or restart your computer.
# (The alternative, chromadb.Client(), only keeps data in memory - gone on exit.)
client = chromadb.PersistentClient(path="./data/chroma")

# A "collection" is like a table in a regular database -
# a named group of related vectors. We'll store all our paper chunks here.
collection = client.get_or_create_collection(name="papers")