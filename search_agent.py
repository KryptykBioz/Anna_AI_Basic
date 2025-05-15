import re
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import ollama  # pip install ollama

# --- Configuration ---
DUCK_URL        = "https://html.duckduckgo.com/html/"
HEADERS         = {"User-Agent": "Mozilla/5.0 (compatible; MySearchBot/1.0)"}
SEARCH_KEYWORDS = [
    r"\b(internet|search|find|latest|current|recent|breaking|new|news|best|greatest|these days)\b",
    r"\b(happening now|right now|just happened)\b",
    r"\b(trending|weather in|price of)\b",
    r"\b(update|live updates)\b",
]
KEYWORD_PATTERN = re.compile("|".join(SEARCH_KEYWORDS), re.IGNORECASE)

AGENT_SYSTEM_PROMPT = (
    "You are a meta-agent whose sole job is to answer YES or NO: "
    "Should the user’s query be answered by fetching fresh web results?"
)

# --- Agent decision ---
def should_search_with_agent(prompt: str) -> bool:
    if "internet" in prompt or "search" in prompt:
        return True
    try:
        resp = ollama.chat(
            model="qwen2.5:3b-instruct-q4_K_M",
            messages=[
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                {"role": "user",   "content": prompt}
            ],
            temperature=0.0,
            max_tokens=4
        )
        ans = resp["choices"][0]["message"]["content"].strip().lower()
        if ans in {"yes", "no"}:
            return ans == "yes"
    except Exception:
        pass
    determination = bool(KEYWORD_PATTERN.search(prompt))
    print(f"SEARCH DETERMINED : {determination}")
    return bool(KEYWORD_PATTERN.search(prompt))
    # return True

# --- Perform the search (titles + URLs) ---
def perform_search(query: str, max_results: int = 5) -> list[dict]:
    # 1) API-based
    try:
        with DDGS() as ddgs:
            hits = ddgs.text(query, max_results=max_results)
            out = [
                {"title": h["title"], "url": h["href"]}
                for h in hits if "title" in h and "href" in h
            ]
            if out:
                return out
    except Exception:
        pass

    # 2) HTML-scrape fallback
    resp = requests.get(DUCK_URL, headers=HEADERS, params={"q": query}, timeout=5)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    results = []
    for a in soup.select("a.result__a")[:max_results]:
        results.append({
            "title": a.get_text(strip=True),
            "url":   a["href"]
        })
    return results


# --- Fetch and extract article text from a URL ---
def fetch_article_text(url: str, timeout: int = 5) -> str:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Try <article> first
        article = soup.find("article")
        if article:
            paragraphs = article.find_all("p")
        else:
            # fallback: all <p> in body
            body = soup.find("body") or soup
            paragraphs = body.find_all("p")

        text = "\n\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
        return text or "[No extractable text found]"
    except Exception as e:
        return f"[Error fetching article: {e}]"


# --- Combine search + fetch article text ---
def maybe_fetch_articles(prompt: str, max_results: int = 3) -> list[dict]:
    """
    If agent (or regex) says YES, search for URLs, then fetch & return
    article text for each. Otherwise return empty list.
    """
    if not should_search_with_agent(prompt):
        return []

    query = prompt.strip().rstrip("?")
    hits  = perform_search(query, max_results=max_results)

    articles = []
    for hit in hits:
        text = fetch_article_text(hit["url"])
        articles.append({
            "title": hit["title"],
            "url":   hit["url"],
            "text":  text
        })
    return articles


# --- Example CLI usage ---
if __name__ == "__main__":
    user_input = input("Ask me anything: ")
    articles   = maybe_fetch_articles(user_input)

    if not articles:
        print("\n[No web search needed or no results]")
    else:
        for idx, art in enumerate(articles, 1):
            print(f"\n--- Article {idx}: {art['title']} ---")
            print(art["url"])
            print(art["text"][:2000])  # print up to first 2000 chars
