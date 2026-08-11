"""
Free, no-key Semantic Scholar search for grounding the Novelty Agent.

Rate limits apply on the unauthenticated tier (roughly 100 requests / 5 min
as of writing) -- fine for dev/testing and the eval batch, but add a small
sleep if you're hammering it in a loop.
"""

import time
import requests

BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"


def search_related_papers(query: str, limit: int = 5, retries: int = 2) -> list[dict]:
    """
    Returns a list of {title, abstract, year} for papers related to `query`
    (typically the input paper's title, or title + a few keywords from the abstract).
    """
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,abstract,year",
    }
    for attempt in range(retries + 1):
        try:
            resp = requests.get(BASE_URL, params=params, timeout=10)
            if resp.status_code == 429:
                time.sleep(2 * (attempt + 1))
                continue
            resp.raise_for_status()
            data = resp.json()
            papers = data.get("data", [])
            return [
                {
                    "title": p.get("title", ""),
                    "abstract": p.get("abstract") or "(no abstract available)",
                    "year": p.get("year"),
                }
                for p in papers
                if p.get("title")
            ]
        except requests.RequestException:
            if attempt == retries:
                return []  # fail gracefully -- novelty agent should handle empty context
            time.sleep(1)
    return []
