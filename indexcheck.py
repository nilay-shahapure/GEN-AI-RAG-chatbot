import pickle, faiss, numpy as np
index = faiss.read_index("vector.index")
print("index size:", index.ntotal, " dim:", index.d)     # expect >0 rows

# grab first 10 rows and check finiteness
xb = faiss.vector_float_to_array(index.reconstruct_n(0, min(10, index.ntotal)))
xb = xb.reshape(-1, index.d)
print("any NaN left?", np.isnan(xb).any())
