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