#!/usr/bin/env python3
"""
google_grounded_hf.py
---------------------

Ground a Hugging Face LLM's answer in fresh Google search results.

Steps
1.  Read a question from the CLI (or prompt for it).
2.  Call Google Custom Search JSON API for the top N web results.
3.  Build a prompt listing those results with numbered citations.
4.  Send that prompt to a Hugging Face chat model (via Inference API or local).
5.  Print the answer and the numbered source list.

Install
-------
pip install requests huggingface_hub python-dotenv

Environment variables
---------------------
GOOGLE_API_KEY   # Google API key
GOOGLE_CX        # Programmable Search Engine ID (“cx”)
HF_MODEL         # e.g.  mistralai/Mixtral-8x7B-Instruct-v0.1
HF_TOKEN         # HF access token (needed for hosted Inference API)
MAX_RESULTS      # optional, default 6
"""

from __future__ import annotations
import os, sys, textwrap, requests
from dotenv import load_dotenv
from browser import browser_search

load_dotenv()

# ---------------------------------------------------------------- helpers
def getenv(name: str) -> str:
    val = os.getenv(name)
    if not val:
        sys.exit(f"✖  Set the {name} environment variable.")
    return val


# ---------------------------------------------------------------- 1. get question
question = " ".join(sys.argv[1:]).strip() or input("Enter your question: ").strip()
if not question:
    sys.exit("No question provided.")

# search question:
print(browser_search(question))


# prompt = f"""<s>[INST]
# You are a concise assistant. Use ONLY the web results below to answer the user's
# question, and cite sources by their list number in square brackets—[1], [2], etc.
# If the information is insufficient, say so.

# User question: {question}

# Web results:
# {context_block}
# [/INST]"""

# # ---------------------------------------------------------------- 4. Hugging Face LLM
# hf_model = os.getenv("HF_MODEL", "mistralai/Mixtral-8x7B-Instruct-v0.1")
# hf_token = os.getenv("HF_TOKEN")               # needed for hosted models
# client   = InferenceClient(model=hf_model, token=hf_token)

# answer = client.text_generation(
#     prompt,
#     max_new_tokens=512,
#     temperature=0.3,
#     top_p=0.9,
#     stop_sequences=["</s>"],
# ).strip()

# # ---------------------------------------------------------------- 5. display answer & sources
# print("\n— Answer —\n")
# print(textwrap.fill(answer, width=100), "\n")

# print("Sources:")
# for idx, (_, link, _) in enumerate(results, 1):
#     print(f"[{idx}] {link}")


# <script async src="https://cse.google.com/cse.js?cx=b58d8f99c7ef746d1">
# </script>
# <div class="gcse-search"></div>
