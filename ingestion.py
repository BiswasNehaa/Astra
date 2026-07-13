import arxiv
from vectorstore import add_chunk

def fetch_papers(query: str, max_results: int=5):
    # Search arXiv for papers matching the query, sorted by newest first.
    client= arxiv.Client()
    
    search= arxiv.Search(
        query= query,
        max_results= max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    
    
     # client.results(search) is what actually sends the request and fetches results
    papers = list(client.results(search))

    return papers

def chunk_text(text: str, chunk_size: int = 100, overlap: int = 20):
    # Split the text into individual words
    words = text.split()

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        # Join this slice of words back into a text chunk
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        # Move the window forward, but overlap slightly with the previous chunk
        # so we don't accidentally cut a sentence in half between chunks.
        start = end - overlap

    return chunks

def ingest_papers(query:str, max_results: int=5):
    # Step 1: fetch papers matching the topic
    papers = fetch_papers(query, max_results)

    total_chunks_saved = 0

    # Step 2: go through each paper one at a time
    for paper in papers:
        # Step 3: break this paper's summary into smaller chunks
        chunks = chunk_text(paper.summary, chunk_size=100, overlap=20)

        # Step 4: save each chunk separately
        for i, chunk in enumerate(chunks):
            # Build a unique ID using the paper's arXiv ID + chunk position.
            # e.g. "2211.13503v1_chunk_0", "2211.13503v1_chunk_1", etc.
            paper_id = paper.entry_id.split("/")[-1]
            chunk_id = f"{paper_id}_chunk_{i}"

            # Metadata helps us know where this text came from later.
            metadata = {
                "title": paper.title,
                "url": paper.entry_id,
                "chunk_index": i,
            }

            add_chunk(chunk_id=chunk_id, text=chunk, metadata=metadata)
            total_chunks_saved += 1

    return total_chunks_saved