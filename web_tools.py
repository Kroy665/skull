"""Web search and page scraping helpers.

web_search() uses DuckDuckGo's no-JS HTML endpoint (no API key required).
scrape_page() fetches a URL and extracts readable text via BeautifulSoup.
"""

import re
from urllib.parse import unquote, urlparse, parse_qs

import requests
from bs4 import BeautifulSoup

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
REQUEST_HEADERS = {"User-Agent": USER_AGENT}

DDG_HTML_URL = "https://html.duckduckgo.com/html/"
MAX_PAGE_BYTES = 2_000_000  # 2MB cap so a huge page doesn't blow up the context


def _clean_ddg_href(href: str) -> str:
    """DuckDuckGo's HTML endpoint wraps result links in a redirect
    (//duckduckgo.com/l/?uddg=<encoded-url>&...); unwrap to the real URL."""
    if href.startswith("//duckduckgo.com/l/") or "duckduckgo.com/l/" in href:
        parsed = urlparse("https:" + href if href.startswith("//") else href)
        qs = parse_qs(parsed.query)
        if "uddg" in qs:
            return unquote(qs["uddg"][0])
    return href


def web_search(query: str, count: int = 5) -> dict:
    if not query:
        return {"error": "Missing required parameter: query"}
    count = max(1, min(int(count), 15))

    try:
        resp = requests.post(
            DDG_HTML_URL,
            data={"q": query},
            headers=REQUEST_HEADERS,
            timeout=20,
        )
    except requests.RequestException as e:
        return {"error": f"Search request failed: {e}"}

    if resp.status_code != 200:
        return {"error": f"Search returned HTTP {resp.status_code}"}

    soup = BeautifulSoup(resp.text, "html.parser")
    results = []
    for result in soup.select(".result"):
        if "result--ad" in (result.get("class") or []):
            continue
        link_el = result.select_one(".result__a")
        snippet_el = result.select_one(".result__snippet")
        if not link_el:
            continue
        url = _clean_ddg_href(link_el.get("href", ""))
        title = link_el.get_text(strip=True)
        snippet = snippet_el.get_text(strip=True) if snippet_el else ""
        # Skip anything that isn't a clean organic link (ad redirects, etc.)
        if not url or not url.startswith("http") or "duckduckgo.com/y.js" in url:
            continue
        if title:
            results.append({"title": title, "url": url, "snippet": snippet})
        if len(results) >= count:
            break

    return {"query": query, "results": results}


def scrape_page(url: str, max_chars: int = 5000) -> dict:
    if not url:
        return {"error": "Missing required parameter: url"}
    if not re.match(r"^https?://", url):
        url = "https://" + url
    max_chars = max(200, min(int(max_chars), 20000))

    try:
        resp = requests.get(
            url,
            headers=REQUEST_HEADERS,
            timeout=20,
            stream=True,
        )
        content = resp.raw.read(MAX_PAGE_BYTES + 1, decode_content=True)
    except requests.RequestException as e:
        return {"error": f"Failed to fetch page: {e}"}

    if resp.status_code != 200:
        return {"error": f"Page returned HTTP {resp.status_code}", "url": url}

    truncated_fetch = len(content) > MAX_PAGE_BYTES
    content = content[:MAX_PAGE_BYTES]

    content_type = resp.headers.get("Content-Type", "")
    if "text/html" not in content_type and "application/xhtml" not in content_type:
        return {
            "error": f"Unsupported content type for scraping: '{content_type or 'unknown'}'",
            "url": url,
        }

    soup = BeautifulSoup(content, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "header"]):
        tag.decompose()

    title = soup.title.get_text(strip=True) if soup.title else ""
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)

    truncated_text = len(text) > max_chars
    text = text[:max_chars]

    return {
        "url": url,
        "title": title,
        "text": text,
        "truncated": truncated_fetch or truncated_text,
    }
