# agents.py
import os, pickle, faiss, numpy as np, re
from duckduckgo_search import DDGS
import requests, urllib.parse 
from dotenv import load_dotenv
load_dotenv()

# ─── load vector index & chunk metadata ─────────────────────────────
index = faiss.read_index("vector.index")
with open("chunks.pkl", "rb") as f:
    data    = pickle.load(f)          # {"texts": …, "sources": …, "pages": …}
    CHUNKS  = data["texts"]
    SOURCES = data["sources"]
    PAGES   = data["pages"]

# ─── OpenAI embeddings helper ───────────────────────────────────────
from agno.embedder.openai import OpenAIEmbedder
EMBEDDER = OpenAIEmbedder(api_key=os.getenv("OPENAI_API_KEY"))

def pdf_search(query: str, k: int = 4) -> str:
    """Return top-k PDF chunks with filename & page tags."""
    vec = EMBEDDER.get_embedding([query])
    if not vec:
        return "[PDF search skipped – empty embedding]"
    q_vec = np.asarray(vec, dtype="float32")
    if q_vec.ndim == 1:
        q_vec = q_vec.reshape(1, -1)
    if q_vec.shape[-1] != index.d:
        fixed = np.zeros((1, index.d), dtype="float32")
        fixed[0, : min(index.d, q_vec.shape[-1])] = q_vec[0, : min(index.d, q_vec.shape[-1])]
        q_vec = fixed
    faiss.normalize_L2(q_vec)
    D, I = index.search(q_vec, k)
    return "\n\n".join(
        f"(score {D[0][j]:.3f}) [{SOURCES[idx]} p{PAGES[idx]}] {CHUNKS[idx]}"
        for j, idx in enumerate(I[0])
    )

# ─── DuckDuckGo Web search tool ─────────────────────────────────────

from itertools import islice

from duckduckgo_search import DDGS         

def web_search(query: str, k: int = 5) -> tuple[str, str | None]:
    """
    Uses DuckDuckGo Instant-Answer JSON API (no key, no ratelimit).
    Returns (joined_context, first_url_or_None)
    """
    api = (
        "https://api.duckduckgo.com/"
        f"?q={urllib.parse.quote_plus(query)}"
        "&format=json&no_redirect=1&no_html=1&skip_disambig=1"
    )
    try:
        data = requests.get(api, timeout=8).json()
    except Exception:
        return "[Web search failed]", None

    hits: list[dict] = []

    # 1) AbstractURL is DDG's primary answer (often Wikipedia)
    if data.get("AbstractURL"):
        hits.append({"title": data.get("Heading") or query,
                     "href":  data["AbstractURL"]})

    # 2) RelatedTopics list
    for item in data.get("RelatedTopics", []):
        # can be a topic or a group of topics
        if "FirstURL" in item:
            hits.append({"title": item.get("Text"), "href": item["FirstURL"]})
        elif "Topics" in item:
            for sub in item["Topics"]:
                if "FirstURL" in sub:
                    hits.append({"title": sub.get("Text"), "href": sub["FirstURL"]})
        if len(hits) >= k:
            break

    if not hits:
        return "[No web results]", None

    # build context string
    ctx = "\n".join(f"{h['title']}: {h['href']}" for h in hits)

    # normalise first URL
    url = hits[0]["href"]
    if not url.startswith(("http://", "https://")):
        url = "https://" + url.lstrip("/")

    return ctx, url



# ─── finance / non-finance classifier (Agno agent) ──────────────────
from agno.models.openai.chat import OpenAIChat
from agno.agent import Agent

LLM = OpenAIChat(id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
classifier = Agent(
    model=LLM,
    instructions=(
        "Return YES if the question is primarily about finance, money, "
        "investing, insurance, loans, retirement, economics. Otherwise NO. "
        "Respond with YES or NO only."
    ),
    markdown=False,
)

def finance_classifier(q: str) -> str:
    return classifier.run(q).content.strip().upper()

# ─── smart context builder with page provenance ─────────────────────
SIM_TH = 0.35

def retrieve_context(q: str, k_pdf: int = 4, k_web: int = 5) -> str:
    """
    • If classifier says NO → web only.
    • If YES → PDF first; keep PDF when score ≥ SIM_TH, else merge with Web.
    Each returned context ends with a single 'Source:' line.
    """

    # 0. finance / non-finance gate
    if finance_classifier(q) == "NO":
        web_ctx, first_url = web_search(q, k_web)
        first_url = first_url or "Web-unavailable"
        return web_ctx + f"\n\nSource: {first_url}"

    # 1. PDF search
    pdf_ctx = pdf_search(q, k=k_pdf)
    m = re.search(r"\[([^\]]+)\s+p(\d+)\].*?\(score ([0-9.]+)\)", pdf_ctx, re.S)
    pdf_name, pdf_page, top_sim = (m.group(1), m.group(2), float(m.group(3))) if m else (None, None, 0.0)

    if pdf_name and top_sim >= SIM_TH:
        return pdf_ctx + f"\n\nSource: {pdf_name} p{pdf_page}"

    # 2. Web fallback (merge PDF + Web)
    web_ctx, first_url = web_search(q, k_web)
    first_url = first_url or "Web-unavailable"

    src_line = (
        f"Source: {pdf_name} p{pdf_page} & {first_url}"
        if pdf_name else f"Source: {first_url}"
    )
    return pdf_ctx + "\n\n" + web_ctx + "\n\n" + src_line



# ─── final RAG agent ------------------------------------------------
rag_agent = Agent(
    model=LLM,
    tools=[retrieve_context],
    instructions=(
        "Call retrieve_context(question) exactly once. "
        "Use ONLY the text it returns to craft your answer. "
        "**Repeat the final 'Source: …' line verbatim at the end.**"
    ),
    markdown=True,
)

def answer(q: str) -> str:
    return rag_agent.run(q).content

# ─── simple CLI loop -----------------------------------------------
if __name__ == "__main__":
    while True:
        q = input("Ask (or 'quit'): ").strip()
        if not q or q.lower() in {"quit", "exit"}:
            break
        print(answer(q))
