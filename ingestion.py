import arxiv

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