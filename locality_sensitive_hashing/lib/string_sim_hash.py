import numpy as np
import pandas as pd
from jarowinkler import jarowinkler_similarity
from rapidfuzz.process import cdist
import faiss


def get_embeddings_simvec(items, compare_items, metric=jarowinkler_similarity):
    """
    Get similarity vector between items and compare_items
    @param items: list of strings
    @param compare_items: list of strings
    @param metric: similarity metric
    @return: similarity vector
    """
    if compare_items is None:
        compare_items = np.random.choice(items, 128)

    embeddings = cdist(items, compare_items, scorer=metric, workers=1)
    return embeddings


def create_faiss_index(embeddings):
    """
    Create FAISS index
    @param embeddings: embeddings
    @return: FAISS index
    """
    ## Create index
    dim = embeddings.shape[1]
    quantizer = faiss.IndexFlatL2(dim)

    n_centroids = 8
    n_bits = 8
    n_vornoi_cells = 2 ** n_bits

    index = faiss.IndexIVFPQ(quantizer, dim, n_vornoi_cells, n_centroids, n_bits)
    index.nprobe = int(np.sqrt(n_vornoi_cells))

    """
    ## Add HNSW
    hnsw_index = faiss.IndexHNSWFlat(dim, 32)
    hnsw_index.hnsw.efSearch = 64
    hnsw_index.hnsw.efConstruction = 64
    index.hnsw_index = hnsw_index
    """

    index.train(embeddings)
    index.add(embeddings)
    '''
    index = quantizer
    index.add(embeddings)
    '''
    return index


def dedupe_faiss(items, cutoff=0.1, k=10):
    """
    Identify duplicate items 
    @param items: list of strings
    @param cutoff: cutoff for similarity
    @param k: number of nearest neighbours
    @return: dataframe of matches
    """
    items = np.array(items)
    compare_items = np.random.choice(items, 128)
    embeddings = get_embeddings_simvec(items, compare_items, metric=jarowinkler_similarity)

    """
    with mp.Pool(os.cpu_count()) as pool:
        embeddings = pool.starmap(
                get_embeddings_simvec, 
                [(items, compare_items, jarowinkler_similarity)]
                )
    """
    embeddings = np.array(embeddings).squeeze()

    ## Normalize embeddings
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, None]
    
    index = create_faiss_index(embeddings)

    ## Get k nearest neighbours
    distances, indices = index.search(embeddings, k)

    match_df = pd.DataFrame({
        'orig_idxs': np.arange(k * len(items)) // k,
        'match_idxs': indices.flatten(),
        'distance': distances.flatten()
    })
    ## Get unique edges and non-self edges
    match_df = match_df[match_df['orig_idxs'] != match_df['match_idxs']]
    match_df = match_df[match_df['orig_idxs'] < match_df['match_idxs']]
    match_df = match_df[match_df['distance'] < cutoff]
    match_df = match_df.reset_index(drop=True)
    return match_df
