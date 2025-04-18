import requests
from bs4 import BeautifulSoup            # pip install beautifulsoup4
from urllib.parse import urlparse

def fetch_page_text(url: str, max_chars: int = 4000) -> str:
    """Download *url*, strip boilerplate, return at most *max_chars* plain text."""
    headers = {
        # A desktop UA helps bypass simplistic bot filters
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0 Safari/537.36"
        )
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
    except requests.exceptions.RequestException as exc:
        print(f"[warn] couldn’t fetch {url}: {exc}")
        return ""

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove obvious boiler‑plate tags
    for tag in soup(["script", "style", "noscript", "header",
                     "footer", "nav", "aside", "form"]):
        tag.decompose()

    # Grab visible paragraph‑like text
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    text = "\n".join(paragraphs)

    # Fallback if <p> tags are scarce (e.g., docs sites)
    if len(text) < 200:
        text = soup.get_text(" ", strip=True)

    return text[:max_chars]

# print(fetch_page_text("https://weather.com/weather/tenday/l/Pittsburgh+PA?canonicalCityId=2b688109f8f42b180dd7d5d4b689f696"))
