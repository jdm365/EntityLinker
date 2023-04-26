import numpy as np
import faiss


def create_faiss_index(embeddings):
    ## Pad to multiple of 8
    if embeddings.shape[-1] % 8 != 0:
        embeddings = np.pad(
                embeddings, 
                ((0, 0), (0, 8 - embeddings.shape[-1] % 8)),
                'constant', 
                constant_values=0
                )

    index = faiss.index_factory(embeddings.shape[-1], "IVF256,PQ32x8")
    index.train(embeddings)
    index.add(embeddings)

    return index
