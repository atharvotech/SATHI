"""
JARVIS Web Tools — Search, Browse, Fetch
Uses DuckDuckGo for search (no API key needed).
"""

import webbrowser
import requests
import re


def web_search(query: str, max_results: int = 5) -> dict:
    """Search the web using DuckDuckGo. Returns titles, URLs, and snippets."""
    try:
        from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", "")
                })
        return {"status": "ok", "result": results}
    except Exception as e:
        return {"status": "error", "result": f"Search failed: {e}"}


def open_url(url: str) -> dict:
    """Open a URL in the default web browser."""
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        webbrowser.open(url)
        return {"status": "ok", "result": f"Opened {url} in browser"}
    except Exception as e:
        return {"status": "error", "result": f"Failed to open URL: {e}"}


def fetch_page_text(url: str, max_chars: int = 3000) -> dict:
    """Fetch a web page and return its text content (stripped of HTML)."""
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) JARVIS/1.0'}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        # Strip HTML tags
        text = re.sub(r'<script[^>]*>.*?</script>', '', resp.text, flags=re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return {"status": "ok", "result": text[:max_chars]}
    except Exception as e:
        return {"status": "error", "result": f"Fetch failed: {e}"}
