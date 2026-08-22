import arxiv
from vectorstore import add_chunk

def fetch_papers(query: str, max_results: int = 5, sort_by_date: bool = False):
    client = arxiv.Client()

    # Default: sort by relevance (best for general search/ask).
    # sort_by_date=True: sort by newest first (best for "recent research" summaries).
    sort_order = arxiv.SortCriterion.SubmittedDate if sort_by_date else arxiv.SortCriterion.Relevance

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=sort_order,
    )
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

def summarize_paper(paper):
    from llm import ask_ai

    prompt = f"""Summarize this research paper abstract in 2-3 clear sentences,
for someone doing a literature review. Focus on what the paper actually did
and found - not generic filler sentences.

Title: {paper.title}
Abstract: {paper.summary}
"""
    summary = ask_ai(prompt)

    return {
        "title": paper.title,
        "authors": [author.name for author in paper.authors],
        "published": str(paper.published.date()),
        "url": paper.entry_id,
        "summary": summary,
    }
    
def summarize_topic(topic: str, max_results: int = 5):
    from llm import ask_ai

    # Step 1: get the most recent papers on this topic
    papers = fetch_papers(topic, max_results=max_results, sort_by_date=True)

    # Step 2: summarize each paper individually
    paper_summaries = [summarize_paper(paper) for paper in papers]

    # Step 3: build one combined overview across all summaries
    combined_text = "\n\n".join(
        f"{p['title']} ({p['published']}): {p['summary']}"
        for p in paper_summaries
    )

    overview_prompt = f"""Based on these {len(paper_summaries)} paper summaries about "{topic}",
write a short overview (3-5 sentences). If the papers are NOT closely related to each other,
say so explicitly instead of forcing connections between them.

{combined_text}
"""
    overall_summary = ask_ai(overview_prompt)

    return {
        "topic": topic,
        "papers": paper_summaries,
        "overall_summary": overall_summary,
    }