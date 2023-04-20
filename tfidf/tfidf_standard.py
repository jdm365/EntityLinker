import pandas as pd
from tqdm import tqdm
from time import perf_counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MiniBatchKMeans
from datasketch import MinHash, MinHashLSH 
import pynndescent
import faiss
import numpy as np
import numba
from scipy.sparse.csgraph import connected_components


def dedupe_knn(items, k=10):
    """
    Remove duplicates 
    """
    tfidf = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
    vectors = tfidf.fit_transform(items)

    compressed_vectors = compress_vectors(vectors)

    neighbors = NearestNeighbors(n_neighbors=k, n_jobs=-1, algorithm='ball_tree')
    neighbors.fit(compressed_vectors)
    distances, idxs = neighbors.kneighbors(compressed_vectors)
    return create_dedupe_df(items, idxs, distances)


@numba.jit(fastmath=True)
def euclidean_distance(x, y):
    """
    Calculate euclidean distance between two vectors
    """
    return np.sqrt(np.sum((x - y)**2))

@numba.jit(fastmath=True)
def cosine_distance(x, y):
    """
    Calculate cosine distance between two vectors
    """
    return 1 - np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))


def create_dedupe_df(items, idxs, distances):
    deduped_df = pd.DataFrame({
        'items': items, 
        'idxs': idxs.tolist(), 
        'distance': distances.tolist()
        }).reset_index(drop=True)
    exploded_df = deduped_df.explode(['idxs', 'distance'])

    ## Remove self matches
    exploded_df = exploded_df[exploded_df['idxs'] != exploded_df.index]

    exploded_df['neighbor_items'] = items[np.array(exploded_df['idxs'].values, dtype=int)]

    ## Normalize distance
    exploded_df['distance'] = exploded_df['distance'] / exploded_df.groupby('items')['distance'].transform('max')
    return exploded_df[['items', 'neighbor_items', 'distance']]


def dedupe_approx_knn(items, k=10):
    """
    Remove duplicates 
    """
    tfidf = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
    vectors = tfidf.fit_transform(items)

    compressed_vectors = compress_vectors(vectors)

    index = pynndescent.NNDescent(compressed_vectors, metric=cosine_distance, n_neighbors=k)
    idxs, distances = index.query(compressed_vectors, k=k)

    return create_dedupe_df(items, idxs, distances)



def shingle(item, k=[3, 4, 5]):
    """
    Create shingles of size k
    """
    shingles = []
    for i in k:
        shingles += [item[j:j+i].encode('utf8') for j in range(len(item) - i + 1)]
    return shingles


def dedupe_faiss(items, k=10):
    """
    Remove duplicates 
    """
    tfidf = TfidfVectorizer(analyzer='char', ngram_range=(3, 4))
    vectors = tfidf.fit_transform(items)

    compressed_vectors = compress_vectors(vectors)
    '''
    shingled_items = []
    for item in tqdm(items):
        shingled_items.append(shingle(item))

    compressed_vectors = np.stack([x.digest() for x in MinHash.bulk(items, num_perm=64)])

    '''
    ## Pad to multiple of 8
    compressed_vectors = np.pad(
            compressed_vectors, 
            ((0, 0), (0, 8 - compressed_vectors.shape[1] % 8)),
            'constant', 
            constant_values=0
            )

    start = perf_counter()
    m = 8  # number of centroid IDs in final compressed vectors
    bits = 8 # number of bits in each centroid
    nlist = 512#0.8 * len(items)  # number of clusters

    quantizer = faiss.IndexFlatL2(compressed_vectors.shape[1])  # we keep the same L2 distance flat index
    index = faiss.IndexIVFPQ(quantizer, compressed_vectors.shape[1], nlist, m, bits) 
    index.train(compressed_vectors)
    index.nprobe = 32


    index.add(compressed_vectors)
    print('Indexing time: {} seconds'.format(perf_counter() - start))
    distances, idxs = index.search(compressed_vectors, k)
    print('Indexing + Search time: {} seconds'.format(perf_counter() - start))
    return create_dedupe_df(items, idxs, distances)


def dedupe_minibatch_kmeans(items, k=10):
    """
    Remove duplicates 
    """
    '''
    tfidf = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
    vectors = tfidf.fit_transform(items)

    compressed_vectors = compress_vectors(vectors)
    '''
    shingled_items = []
    for item in tqdm(items):
        shingled_items.append(shingle(item))

    compressed_vectors = np.stack([x.digest() for x in MinHash.bulk(shingled_items, num_perm=64)])

    ## Normalize
    compressed_vectors = compressed_vectors / np.linalg.norm(compressed_vectors, axis=1, keepdims=True)

    kmeans = MiniBatchKMeans(n_clusters= 4 * len(compressed_vectors) // 5, batch_size=1024, max_iter=1000, verbose=True)
    clusters = kmeans.fit_predict(compressed_vectors)
    return kmeans.centroids
