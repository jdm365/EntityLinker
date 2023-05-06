import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import pynndescent
import faiss
import numpy as np
import numba
from tqdm import tqdm

from tfidf.lib.vector_compression import compress_vectors



def create_dedupe_df(idxs, distances):
    """
    Create a dataframe of matches from the output of a knn model
    @param idxs: array of indices of matches
    @param distances: array of distances of matches
    @return: dataframe of matches
    """
    deduped_df = pd.DataFrame({
        'orig_idxs': np.arange(len(idxs)),
        'match_idxs': idxs.tolist(), 
        'distance': distances.tolist()
        }).explode(['match_idxs', 'distance'])

    ## Remove self matches
    deduped_df = deduped_df[deduped_df['orig_idxs'] != deduped_df['match_idxs']]
    deduped_df = deduped_df[deduped_df['orig_idxs'] < deduped_df['match_idxs']]
    return deduped_df


def get_compressed_embeddings(items, dim=None, return_tfidf=False):
    """
    Get compressed embeddings from a list of items
    @param items: list of items
    @param dim: dimension of compressed embeddings
    @param return_tfidf: whether to return tfidf vectorizer
    @return: compressed embeddings
    """
    vectors, tfidf = get_embeddings(items)
    compressed_vectors = compress_vectors(vectors, n_singular_values=dim)
    if return_tfidf:
        return compressed_vectors, tfidf
    return compressed_vectors


def get_embeddings(items):
    """
    Get embeddings from a list of items
    @param items: list of items
    @return: (embeddings, tfidf vectorizer)
    """
    tfidf = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))
    vectors = tfidf.fit_transform(items)

    return vectors, tfidf


def dedupe_knn(items, k=10):
    """
    Remove duplicates 
    @param items: list of items
    @param k: number of neighbors to consider
    @return: dataframe of matches
    """
    compressed_vectors = get_compressed_embeddings(items, dim=64)

    neighbors = NearestNeighbors(n_neighbors=k, n_jobs=-1, algorithm='kd_tree')
    neighbors.fit(compressed_vectors)
    distances, idxs = neighbors.kneighbors(compressed_vectors)
    return create_dedupe_df(idxs, distances)


@numba.jit(fastmath=True)
def euclidean_distance(x, y):
    """
    Calculate euclidean distance between two vectors
    @param x: vector
    @param y: vector
    @return: euclidean distance
    """
    return np.sqrt(np.sum((x - y) ** 2))


@numba.jit(fastmath=True)
def cosine_distance(x, y):
    """
    Calculate cosine distance between two vectors
    @param x: vector
    @param y: vector
    @return: cosine distance
    """
    return 1 - np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))


def get_approx_knn_index(embeddings, k=10):
    """
    Create an approximate knn index
    @param embeddings: embeddings
    @param k: number of neighbors to consider
    @return: approximate knn index
    """
    index = pynndescent.NNDescent(embeddings, metric=euclidean_distance, n_neighbors=k)
    return index


def dedupe_approx_knn(items, k=10):
    """
    Remove duplicates 
    @param items: list of items
    @param k: number of neighbors to consider
    @return: dataframe of matches
    """
    compressed_vectors = get_compressed_embeddings(items, dim=64)

    index = get_approx_knn_index(compressed_vectors, k=k)
    idxs, distances = index.query(compressed_vectors, k=k)

    return create_dedupe_df(idxs, distances)


def create_faiss_index(embeddings):
    ## Pad to multiple of 8
    if embeddings.shape[-1] % 8 != 0:
        embeddings = np.pad(
                embeddings, 
                ((0, 0), (0, 8 - embeddings.shape[-1] % 8)),
                'constant', 
                constant_values=0
                )
    if embeddings.shape[0] > 50_000:
        index = faiss.index_factory(embeddings.shape[-1], "IVF512,PQ32")
        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = 32
    else:
        index = faiss.IndexFlatL2(embeddings.shape[-1])
        index.add(embeddings)

    return index, embeddings


def dedupe_faiss(items, k=5):
    """
    Remove duplicates 
    @param items: list of items
    @param k: number of neighbors to consider
    @return: dataframe of matches
    """
    compressed_vectors = get_compressed_embeddings(items)
    index, compressed_vectors = create_faiss_index(compressed_vectors)

    distances, idxs = index.search(compressed_vectors, k)
    return create_dedupe_df(idxs, distances)


def dedupe_bm25(items, k=5):
    """
    Remove duplicates 
    @param items: list of items
    @param k: number of neighbors to consider
    @return: dataframe of matches
    """
    raise NotImplementedError

    ## Shingle each item to 2-4 ngams
    shingled_items = []
    for item in tqdm(items):
        items_shingled = []
        for size in [2, 3, 4]:
            items_shingled.append(" ".join([item[idx:idx + size] for idx in range(len(item) - size + 1)]))
           
        shingled_items.append(items_shingled)


    all_distances = []
    all_idxs = []
    bm25 = BM25Okapi(shingled_items)
    for item in tqdm(shingled_items):
        distances = bm25.get_scores(item)
        if len(distances) == 0:
            continue
        idxs = np.argsort(distances)[-k:]

        all_distances.append(distances)
        all_idxs.append(idxs)

    distances = np.concatenate(distances)
    idxs = np.concatenate(idxs)
    return create_dedupe_df(idxs, distances)
