# build_index.py  – run whenever “Documents/” changes
# --------------------------------------------------------------------
import os, pathlib, pickle, numpy as np, faiss
from dotenv import load_dotenv

load_dotenv()                                    # pulls OPENAI_API_KEY
from PyPDF2 import PdfReader
from agno.document import Document
from agno.document.chunking.recursive import RecursiveChunking
from agno.embedder.openai import OpenAIEmbedder

PDF_DIR    = pathlib.Path("Documents")
INDEX_FILE = "vector.index"
CHUNK_FILE = "chunks.pkl"

CHUNK_SIZE = 800
OVERLAP    = 50
BATCH      = 96

# ───── 1. read PDFs page-by-page ────────────────────────────────────
print("Importing PDFs …")
pdf_pages, pdf_names, page_nums = [], [], []      # parallel lists
for pdf_path in PDF_DIR.glob("*.pdf"):
    reader = PdfReader(str(pdf_path))
    for idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pdf_pages.append(text)
        pdf_names.append(pdf_path.name)
        page_nums.append(idx)
print(f"  Loaded {len(set(pdf_names))} PDF files • {len(pdf_pages)} pages")

# ───── 2. chunk each page & track filename + page no. ───────────────
chunker = RecursiveChunking(chunk_size=CHUNK_SIZE, overlap=OVERLAP)

chunks, sources, pages = [], [], []
for text, name, pg in zip(pdf_pages, pdf_names, page_nums):
    for sub in chunker.chunk(Document(content=text)):
        chunks.append(sub.content)
        sources.append(name)
        pages.append(pg)
print(f"  Produced {len(chunks)} chunks")

# ───── 3. embed chunks one-by-one (OpenAI API) ──────────────────────
embedder = OpenAIEmbedder(api_key=os.getenv("OPENAI_API_KEY"))
vec_rows = []
for i in range(0, len(chunks), BATCH):
    for txt in chunks[i : i + BATCH]:
        vec_rows.append(embedder.get_embedding(txt))        # 1×1536 each
vectors = np.asarray(vec_rows, dtype="float32")
print("  Embedded:", vectors.shape)

# ───── 3b. drop NaN / Inf / zero-norm rows ‐ keep alignment ─────────
good = np.isfinite(vectors).all(axis=1)
vectors, chunks = vectors[good], [c for c, k in zip(chunks, good) if k]
sources = [s for s, k in zip(sources, good) if k]
pages   = [p for p, k in zip(pages,   good) if k]

faiss.normalize_L2(vectors)
good2 = np.isfinite(vectors).all(axis=1) & (np.linalg.norm(vectors, 1) > 1e-5)
drop  = len(good) - np.count_nonzero(good2)

vectors = vectors[good2]
chunks  = [c for c, k in zip(chunks,  good2) if k]
sources = [s for s, k in zip(sources, good2) if k]
pages   = [p for p, k in zip(pages,   good2) if k]

print(f"  Filtered out {drop} bad chunks • final matrix {vectors.shape}")

# ───── 4. build FAISS index (cosine via normalised IP) ──────────────
index = faiss.IndexFlatIP(vectors.shape[1])
index.add(vectors)
faiss.write_index(index, INDEX_FILE)

# ───── 5. save pickle with text + filename + page no. ───────────────
with open(CHUNK_FILE, "wb") as f:
    pickle.dump({"texts": chunks, "sources": sources, "pages": pages}, f)

print(f"✓ Saved FAISS index → {INDEX_FILE}")
print(f"✓ Saved chunks/sources/pages → {CHUNK_FILE}")
