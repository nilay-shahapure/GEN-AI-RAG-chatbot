# build_index.py  – run whenever the Documents/ folder changes

import os, pathlib, pickle, numpy as np, faiss
from dotenv import load_dotenv

load_dotenv()                       # api key given by shrimukh sir is pulled

from PyPDF2 import PdfReader
from agno.document import Document
from agno.document.chunking.recursive import RecursiveChunking
from agno.embedder.openai import OpenAIEmbedder

# -------- configuration ---------------------------------------------
PDF_DIR    = pathlib.Path("Documents")
INDEX_FILE = "vector.index"
CHUNK_FILE = "chunks.pkl"

CHUNK_SIZE = 800
OVERLAP    = 50
BATCH      = 96

# -------- 1. read all PDFs ------------------------------------------
print("Importing PDFs …")
pdf_texts, pdf_names = [], []
for pdf_path in PDF_DIR.glob("*.pdf"):
    text = "\n".join(
        page.extract_text() or "" for page in PdfReader(str(pdf_path)).pages
    )
    pdf_texts.append(text)
    pdf_names.append(pdf_path.name)
print(f"  Loaded {len(pdf_texts)} PDF files")

# -------- 2. chunk and remember source ------------------------------
chunker = RecursiveChunking(chunk_size=CHUNK_SIZE, overlap=OVERLAP)
chunks, sources = [], []

for text, name in zip(pdf_texts, pdf_names):
    for sub in chunker.chunk(Document(content=text)):
        chunks.append(sub.content)
        sources.append(name)               # same index as its chunk
print(f"  Produced {len(chunks)} chunks")

# ────────── 3. embed all chunks CORRECTLY ───────────────────────────
embedder = OpenAIEmbedder(api_key=os.getenv("OPENAI_API_KEY"))

vectors = []
for i in range(0, len(chunks), BATCH):
    for text in chunks[i : i + BATCH]:          # ← embed 1-by-1
        vec = embedder.get_embedding(text)      # list[1536]
        vectors.append(vec)

vectors = np.asarray(vectors, dtype="float32")  # shape (N_chunks, 1536)
print("  Embedded all chunks:", vectors.shape)  # should now read (443, 1536)


# -------- 3b. drop NaN / Inf / zero-norm rows -----------------------
finite_mask = np.isfinite(vectors).all(axis=1)
vectors     = vectors[finite_mask]
chunks      = [c for c, keep in zip(chunks,  finite_mask) if keep]
sources     = [s for s, keep in zip(sources, finite_mask) if keep]

faiss.normalize_L2(vectors)               # may create NaN on zero vectors
good_mask = np.isfinite(vectors).all(axis=1) & (np.linalg.norm(vectors, 1) > 1e-5)
drop_cnt  = len(finite_mask) - np.count_nonzero(good_mask)

vectors = vectors[good_mask]
chunks  = [c for c, k in zip(chunks,  good_mask) if k]
sources = [s for s, k in zip(sources, good_mask) if k]

print(f"  Filtered out {drop_cnt} bad chunks (NaN / Inf / zero-norm)")
print("  Final matrix shape:", vectors.shape)

# -------- 4. build FAISS index --------------------------------------
index = faiss.IndexFlatIP(vectors.shape[1])   # cosine via L2-normalised IP
index.add(vectors)
faiss.write_index(index, INDEX_FILE)

# -------- 5. save chunks + sources ----------------------------------
with open(CHUNK_FILE, "wb") as f:
    pickle.dump({"texts": chunks, "sources": sources}, f)

print(f"✓ Saved FAISS index → {INDEX_FILE}")
print(f"✓ Saved chunks & sources → {CHUNK_FILE}")
