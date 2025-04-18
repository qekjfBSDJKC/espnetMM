from __future__ import annotations
import os, sys, textwrap, requests
from dotenv import load_dotenv
import fetch_page_text

load_dotenv()

# ---------------------------------------------------------------- helpers
def getenv(name: str) -> str:
    val = os.getenv(name)
    if not val:
        sys.exit(f"✖  Set the {name} environment variable.")
    return val

def search(query, num_links = 2):
    # ---------------------------------------------------------------- 2. Google search
    api_key = getenv("GOOGLE_API_KEY")
    cx      = getenv("GOOGLE_CX")

    resp = requests.get(
        "https://www.googleapis.com/customsearch/v1",
        params={"key": api_key, "cx": cx, "q": query, "num": num_links, "hl": "en"},
        timeout=10,
    )
    resp.raise_for_status()
    items = resp.json().get("items", [])
    if not items:
        sys.exit("Google search returned no items (quota exhausted or query too narrow).")

    results = [(it["title"], it["link"], it.get("snippet", "").replace("\n", " "))
            for it in items]
    return results

def format_results(res):
    out = []
    for i, (title, link, snippet) in enumerate(res, 1):
        out.append(f"{i}. {title}\n   URL: {link}\n   Snippet: {snippet}")
    return "\n".join(out)

def grab_text_from_link(res, max_chars = 4000):
    out = []
    for i, (title, link, snippet) in enumerate(res, 1):
        out.append(f"link title: {title}; link content: {fetch_page_text.fetch_page_text(link, max_chars)}\n")
    return "\n".join(out)

def browser_search(query, max_chars = 4000, num_links = 2):
    search_res = search(query, num_links)
    print(search_res)
    return grab_text_from_link(search_res, max_chars)