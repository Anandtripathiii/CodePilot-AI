"""The agent: searches the open web for documentation.

Kept deliberately simple — one tool, one job. If the search library is
missing or the network is down it returns an empty list and the app just
answers without web context.
"""

from typing import Any, Final, TypedDict

# The package was renamed from `duckduckgo_search` to `ddgs`. Prefer the
# new name, fall back to the old one so existing installs keep working.
DDGS: Any
try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None

# Results from these hosts are sorted to the top.
DOC_SITES: Final[tuple[str, ...]] = (
    "docs.python.org",
    "developer.mozilla.org",
    "stackoverflow.com",
    "readthedocs.io",
    "github.com",
)


class SearchResult(TypedDict):
    title: str
    snippet: str
    url: str


def search_docs(query: str, max_results: int = 4) -> list[SearchResult]:
    """Search for documentation. Never raises — returns [] on any failure."""
    if DDGS is None:
        return []

    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(f"{query} documentation", max_results=max_results * 2))
    except Exception:
        # Network down, rate limited, upstream markup changed — all the same
        # to us. Web context is optional, so degrade rather than fail.
        return []

    results: list[SearchResult] = []
    for item in raw:
        url = item.get("href") or item.get("url") or ""
        results.append(
            SearchResult(
                title=(item.get("title") or url)[:120],
                snippet=(item.get("body") or "")[:280],
                url=url,
            )
        )

    results.sort(key=lambda r: 0 if any(d in r["url"] for d in DOC_SITES) else 1)
    return results[:max_results]
