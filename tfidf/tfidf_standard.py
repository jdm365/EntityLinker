import pandas as pd
from tqdm import tqdm
from time import perf_counter
from vector_compression import * 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN, MiniBatchKMeans
from datasketch import MinHash, MinHashLSH 
import pynndescent
import faiss
import torch as T
import numpy as np
import sys
from scipy.spatial import KDTree
import logging
import numba
from scipy.sparse.csgraph import connected_components

sys.path.append('../')
from utils.address_handler import get_addresses
from utils.fuzzify import Fuzzifier




def get_similar_docs(addresses, query_address, k=10):
    """
    Get similar documents
    """
    tfidf = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
    vectors = tfidf.fit_transform(addresses)
    query_tfidf = tfidf.transform([query_address])
    similarity = (vectors * query_tfidf.T).A
    match_idxs = similarity.argsort(axis=0)[-k:][::-1].flatten()
    return match_idxs



def dedupe_knn(addresses, k=10):
    """
    Remove duplicate addresses
    """
    tfidf = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
    vectors = tfidf.fit_transform(addresses)

    compressed_vectors = compress_vectors(vectors)

    neighbors = NearestNeighbors(n_neighbors=k, n_jobs=-1, algorithm='ball_tree')
    neighbors.fit(compressed_vectors)
    distances, idxs = neighbors.kneighbors(compressed_vectors)
    return create_dedupe_df(addresses, idxs, distances)


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


def create_dedupe_df(addresses, idxs, distances):
    deduped_df = pd.DataFrame({
        'address': addresses, 
        'idxs': idxs.tolist(), 
        'distance': distances.tolist()
        }).reset_index(drop=True)
    exploded_df = deduped_df.explode(['idxs', 'distance'])

    ## Remove self matches
    exploded_df = exploded_df[exploded_df['idxs'] != exploded_df.index]

    exploded_df['neighbor_address'] = addresses[np.array(exploded_df['idxs'].values, dtype=int)]

    ## Normalize distance
    exploded_df['distance'] = exploded_df['distance'] / exploded_df.groupby('address')['distance'].transform('max')
    return exploded_df[['address', 'neighbor_address', 'distance']]


def dedupe_approx_knn(addresses, k=10):
    """
    Remove duplicate addresses
    """
    tfidf = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
    vectors = tfidf.fit_transform(addresses)

    compressed_vectors = compress_vectors(vectors)

    index = pynndescent.NNDescent(compressed_vectors, metric=cosine_distance, n_neighbors=k)
    idxs, distances = index.query(compressed_vectors, k=k)

    return create_dedupe_df(addresses, idxs, distances)



def shingle(address, k=[3, 4, 5]):
    """
    Create shingles of size k
    """
    shingles = []
    for i in k:
        shingles += [address[j:j+i].encode('utf8') for j in range(len(address) - i + 1)]
    return shingles


def dedupe_faiss(addresses, k=10):
    """
    Remove duplicate addresses
    """
    tfidf = TfidfVectorizer(analyzer='char', ngram_range=(3, 4))
    vectors = tfidf.fit_transform(addresses)

    compressed_vectors = compress_vectors(vectors)
    '''
    shingled_addresses = []
    for address in tqdm(addresses):
        shingled_addresses.append(shingle(address))

    compressed_vectors = np.stack([x.digest() for x in MinHash.bulk(shingled_addresses, num_perm=64)])

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
    nlist = 512#0.8 * len(addresses)  # number of clusters

    quantizer = faiss.IndexFlatL2(compressed_vectors.shape[1])  # we keep the same L2 distance flat index
    index = faiss.IndexIVFPQ(quantizer, compressed_vectors.shape[1], nlist, m, bits) 
    index.train(compressed_vectors)
    index.nprobe = 32


    index.add(compressed_vectors)
    print('Indexing time: {} seconds'.format(perf_counter() - start))
    distances, idxs = index.search(compressed_vectors, k)
    print('Indexing + Search time: {} seconds'.format(perf_counter() - start))
    return create_dedupe_df(addresses, idxs, distances)


def dedupe_faiss_clustering(addresses, k=10):
    """
    Remove duplicate addresses
    """
    '''
    tfidf = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
    vectors = tfidf.fit_transform(addresses)

    compressed_vectors = compress_vectors(vectors)
    '''
    shingled_addresses = []
    for address in tqdm(addresses):
        shingled_addresses.append(shingle(address))

    compressed_vectors = np.stack([x.digest() for x in MinHash.bulk(shingled_addresses, num_perm=64)])

    ## Normalize
    compressed_vectors = compressed_vectors / np.linalg.norm(compressed_vectors, axis=1, keepdims=True)

    start = perf_counter()
    n_centriods = 1024
    n_iters = 20
    verbose = True
    dim = compressed_vectors.shape[1]

    kmeans = faiss.Kmeans(dim, n_centriods, niter=n_iters, verbose=verbose)
    kmeans.train(compressed_vectors)

    print(kmeans.centroids.shape)
    sys.exit()
    return kmeans.centroids


def dedupe_minibatch_kmeans(addresses, k=10):
    """
    Remove duplicate addresses
    """
    '''
    tfidf = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
    vectors = tfidf.fit_transform(addresses)

    compressed_vectors = compress_vectors(vectors)
    '''
    shingled_addresses = []
    for address in tqdm(addresses):
        shingled_addresses.append(shingle(address))

    compressed_vectors = np.stack([x.digest() for x in MinHash.bulk(shingled_addresses, num_perm=64)])

    ## Normalize
    compressed_vectors = compressed_vectors / np.linalg.norm(compressed_vectors, axis=1, keepdims=True)

    start = perf_counter()
    kmeans = MiniBatchKMeans(n_clusters= 4 * len(compressed_vectors) // 5, batch_size=1024, max_iter=1000, verbose=True)
    clusters = kmeans.fit_predict(compressed_vectors)

    print(clusters)
    sys.exit()
    return kmeans.centroids



def dedupe_radius_neighbors_graph(addresses, k=10):
    """
    Remove duplicate addresses
    """
    '''
    tfidf = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
    vectors = tfidf.fit_transform(addresses)

    compressed_vectors = compress_vectors(vectors)
    '''
    shingled_addresses = []
    for address in tqdm(addresses):
        shingled_addresses.append(shingle(address))

    compressed_vectors = np.stack([x.digest() for x in MinHash.bulk(shingled_addresses, num_perm=64)])

    ## Normalize
    compressed_vectors = compressed_vectors / np.linalg.norm(compressed_vectors, axis=1, keepdims=True)

    start = perf_counter()
    model = NearestNeighbors(algorithm='kd_tree', n_jobs=-1).fit(compressed_vectors)
    adj_matrix = model.radius_neighbors_graph(compressed_vectors, radius=0.05, mode='connectivity')

    ## Get clusters connected components from adjacency matrix
    n_subgraphs, labels = connected_components(adj_matrix, directed=False, return_labels=True)

    print('Number of clusters: {}'.format(n_subgraphs))
    print('Labels: {}'.format(labels))
    sys.exit()


def dedupe_lsh(addresses, k=10):
    """
    Remove duplicate addresses
    """
    '''
    tfidf = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
    vectors = tfidf.fit_transform(addresses)

    compressed_vectors = compress_vectors(vectors)
    '''
    shingled_addresses = []
    for address in tqdm(addresses):
        shingled_addresses.append(shingle(address))

    ## MinHashLSH
    lsh = MinHashLSH(threshold=0.8, num_perm=64)
    for idx, address in enumerate(tqdm(shingled_addresses, desc='Inserting into LSH')):
        minhash = MinHash(num_perm=64)
        for d in address:
            minhash.update(d)
        lsh.insert(idx, minhash)

    results = []
    for idx, address in enumerate(tqdm(shingled_addresses, desc='Querying LSH')):
        minhash = MinHash(num_perm=64)
        for d in address:
            minhash.update(d)
        result = lsh.query(minhash)
        results.append(result)

    df = pd.DataFrame({'address': addresses, 'results': results})
    df['cluster_id'] = np.arange(len(df))
    df = df.explode('results')
    df = df[df['results'] != df.index].reset_index(drop=True)
    df['neighbor_address'] = addresses[np.array(df['results'].values, dtype=int)]
    print(df)
    sys.exit()
    return df 




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    data = pd.read_feather('../data/extracted_addresses.feather')
    data = data[data['address'] != '']
    data = data[data['address'].isnull() == False]

    SUBSET_SIZE = 100_000

    addresses = np.random.choice(data['address'], SUBSET_SIZE, replace=False)
    ## Remove nulls
    addresses = addresses

    print('Number of addresses: {}'.format(len(addresses)))

    fuzzifier = Fuzzifier()

    #query_address = addresses[500]
    query_address = fuzzifier.fuzzify(addresses[500])


    start = perf_counter()
    #match_df = get_similar_docs(addresses, query_address)
    match_df = dedupe_faiss(addresses)
    #match_df = dedupe_faiss_clustering(addresses)
    #match_df = dedupe_radius_neighbors_graph(addresses)
    #match_df = dedupe_minibatch_kmeans(addresses)
    #match_df = dedupe_lsh(addresses)
    #match_df = dedupe_approx_knn(addresses)
    end = perf_counter()

    print(match_df[match_df['distance'] < 0.08])

    print('Time taken: {} seconds'.format(end - start))
