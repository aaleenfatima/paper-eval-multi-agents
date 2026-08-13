"""
Semantic Scholar search for grounding the Novelty Agent.

Free, unauthenticated use is heavily rate-limited (a handful of requests per
minute). This version throttles calls to avoid tripping that limit in the
first place, and backs off properly (respecting Retry-After) when it does
happen instead of giving up after one try.

Optional: an API key (free signup, no cost) would raise the rate limit
further -- https://www.semanticscholar.org/product/api#api-key-form -- but
this works without one; set SEMANTIC_SCHOLAR_API_KEY as an environment
variable later if the batch run is still hitting rate limits too often.
"""

import os
import time
import requests

BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip()

# Without a key the free tier is easy to trip. A small delay between calls
# avoids hitting it in the first place rather than just retrying after.
MIN_DELAY_SECONDS = 1.0 if API_KEY else 4.0

_last_call_time = 0.0


def _throttle():
    global _last_call_time
    elapsed = time.time() - _last_call_time
    if elapsed < MIN_DELAY_SECONDS:
        time.sleep(MIN_DELAY_SECONDS - elapsed)
    _last_call_time = time.time()


def search_related_papers(query: str, limit: int = 5, retries: int = 3) -> list[dict]:
    """
    Returns a list of {title, abstract, year} for papers related to `query`.
    Returns [] only after exhausting retries -- and prints a warning when it
    does, so an ungrounded novelty verdict is visible in the logs instead of
    silently indistinguishable from "genuinely no related papers exist."
    """
    headers = {"x-api-key": API_KEY} if API_KEY else {}
    params = {"query": query, "limit": limit, "fields": "title,abstract,year"}

    for attempt in range(retries + 1):
        _throttle()
        try:
            resp = requests.get(BASE_URL, params=params, headers=headers, timeout=10)

            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else (8 * (attempt + 1))
                print(f"    [semantic_scholar] rate limited (429), waiting {wait:.0f}s "
                      f"(attempt {attempt + 1}/{retries + 1})...", flush=True)
                time.sleep(wait)
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
        except requests.RequestException as e:
            print(f"    [semantic_scholar] request failed: {e} "
                  f"(attempt {attempt + 1}/{retries + 1})", flush=True)
            if attempt == retries:
                break
            time.sleep(2 * (attempt + 1))

    print(f"    [semantic_scholar] WARNING: giving up after {retries + 1} attempts -- "
          f"novelty verdict for this paper will be ungrounded. Query was: '{query[:80]}'", flush=True)
    return []