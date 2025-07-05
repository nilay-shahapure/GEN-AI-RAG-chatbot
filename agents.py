# agents.py 

import os, asyncio, pickle, faiss, numpy as np
from dotenv import load_dotenv
load_dotenv()

#loading prebuilt index
index = faiss.read_index("vector.index")

with open("chunks.pkl", "rb") as f:
    data     = pickle.load(f)          # {"texts": [...], "sources": [...]}
    CHUNKS   : list[str]  = data["texts"]
    SOURCES  : list[str]  = data["sources"]


from agno.embedder.openai import OpenAIEmbedder
EMBEDDER = OpenAIEmbedder(api_key=os.getenv("OPENAI_API_KEY"))


def pdf_search(query: str, k: int = 4) -> str:
    vec = EMBEDDER.get_embedding([query])
    if not vec:
        return "[PDF search skipped – empty embedding]"

    q_vec = np.asarray(vec, dtype="float32")

    # --- guarantee q_vec is (1, current_dim) ------------------------
    if q_vec.ndim == 1:                       # shape (d,)  → (1, d)
        q_vec = q_vec.reshape(1, -1)

    # --- pad / truncate to index.d ---------------------------------
    if q_vec.shape[-1] != index.d:
        fixed = np.zeros((1, index.d), dtype="float32")
        copy_dim = min(index.d, q_vec.shape[-1])
        fixed[0, :copy_dim] = q_vec[0, :copy_dim]
        q_vec = fixed                              # now (1, index.d)

    faiss.normalize_L2(q_vec)
    D, I = index.search(q_vec, k)
    return "\n\n".join(
        f"(score {D[0][j]:.3f}) [{SOURCES[idx]}] {CHUNKS[idx]}"
        for j, idx in enumerate(I[0]))






from duckduckgo_search import DDGS
from itertools import islice

def web_search(query: str, k: int = 5) -> str:
    """Return first k DuckDuckGo web results (title + URL)."""
    with DDGS() as ddgs:
        hits = list(islice(ddgs.text(query), k))  
    return "\n".join(f"{h['title']}: {h['href']}" for h in hits)

import re

def retrieve_context(q: str, k_pdf: int = 4, k_web: int = 5) -> str:
    """
    • If the classifier says NO → return Web results only.
    • Else run PDF search first; if similarity ≥ SIM_TH return PDF only,
      otherwise merge PDF & Web.
    """

    # ---------- 0. domain check ------------------------------------
    if finance_classifier(q) == "NO":
        # --- plain Web search --------------------------------------
        with DDGS() as ddgs:
            hits = list(islice(ddgs.text(q), k_web))
        if not hits:
            return "No web results found.\n\nSource: Web"
        first_url = hits[0]["href"]
        web_ctx   = "\n".join(f"{h['title']}: {h['href']}" for h in hits)
        return web_ctx + f"\n\nSource: {first_url}"

    # ---------- 1. PDF search (finance query) ----------------------
    pdf_ctx = pdf_search(q, k=k_pdf)

    m = re.search(r"\[([^\]]+)\].*?\(score ([0-9.]+)\)", pdf_ctx, re.S)
    pdf_name = m.group(1) if m else None
    top_sim  = float(m.group(2)) if m else 0.0

    if pdf_name and top_sim >= SIM_TH:          # strong match
        return pdf_ctx + f"\n\nSource: {pdf_name}"

    # ---------- 2. Web fallback ------------------------------------
    with DDGS() as ddgs:
        hits = list(islice(ddgs.text(q), k_web))
    first_url = hits[0]["href"] if hits else "No-result"
    web_ctx   = "\n".join(f"{h['title']}: {h['href']}" for h in hits)

    src = f"Source: {pdf_name} & {first_url}" if pdf_name else f"Source: {first_url}"
    return pdf_ctx + "\n\n" + web_ctx + "\n\n" + src



#agent for domain classification
from agno.models.openai.chat import OpenAIChat
from agno.agent import Agent

LLM = OpenAIChat(id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

classifier = Agent(
    model=LLM,
    instructions=(
        "Return YES if the question is primarily about finance, money, "
        "banking, investing, retirement (e.g. 401k, IRA), insurance, "
        "loans, interest rates, portfolios or economics. Otherwise NO. "
        "Respond with YES or NO only."
    ),
    markdown=False,
)

def finance_classifier(q: str) -> str:
    return classifier.run(q).content.strip().upper()

#main rag agent section
PROMPT = """
1. Call finance_classifier(question).
2. If the answer is NO → call web_search() and answer from those results.
3. If YES → call pdf_search() to get PDF chunks with scores.
   • If top score ≥ 0.50 → answer using only the PDF chunks.
   • Otherwise call web_search() too and merge both sources.
4. Your answer **must end with one line exactly in this form**  
   Source: PDF            ← if you used only PDF chunks  
   Source: Web - <url>    ← if you used only Web (put the first URL)  
   Source: PDF & Web - <url> ← if you used both.
5. Do not output any other metadata—only the answer and that final Source line.
"""
SIM_TH = 0.25

PROMPT2 = f"""
1. Call finance_classifier(question).
2. If NO → call web_search() and answer from those results.
3. If YES → always call pdf_search() **first** to get chunks with scores.
   • If top score ≥ {SIM_TH} → answer using those PDF chunks.
   • Otherwise call web_search() too and merge both sources.
4. Finish with one Source: … line.
"""

rag_agent = Agent(
    model        = LLM,
    tools        = [retrieve_context],  
    instructions = (
        "Call retrieve_context(question) exactly once. "
        "Use ONLY the text it returns to draft your answer. "
        "**Repeat the final 'Source: …' line verbatim at the end of your answer.**"
    ),
    markdown     = True,
)

def answer(q: str) -> str:
    return rag_agent.run(q).content




#CLI LOOp
if __name__ == "__main__":
    while True:
        query = input("Ask (or 'quit'): ").strip()
        if not query or query.lower() in {"quit", "exit"}:
            break
        print(answer(query))
