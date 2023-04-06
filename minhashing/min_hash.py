import pandas as pd
import numpy as np
from hashlib import sha1
import random
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm
from time import perf_counter
from jarowinkler import jarowinkler_similarity
from rapidfuzz.process import cdist
import faiss
import multiprocessing as mp
import os

import sys
sys.path.append('../')
from utils.address_handler import get_addresses
from utils.fuzzify import Fuzzifier



def create_mh_index(data, num_perm=128, threshold=0.6):
    """
    Create MinHash index from a list of data
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm, hashfunc=hash)
    for i, d in enumerate(tqdm(data, desc='Creating MinHash index')):
        m = MinHash(num_perm=num_perm)
        for s in shingle(d):
            m.update(s.encode('utf8'))
        lsh.insert(i, m)
    return lsh


def get_similar_docs(data, query, num_perm=128, threshold=0.6):
    """
    Get similar documents from a list of data
    """
    lsh = create_mh_index(data, num_perm=num_perm, threshold=threshold)
    m = MinHash(num_perm=num_perm)
    for s in shingle(query):
        m.update(s.encode('utf8'))
    match_idxs = lsh.query(m)
    return match_idxs 


def get_jaccard(s1, s2):
    """
    Get Jaccard similarity between two strings
    """
    s1 = set(s1.split())
    s2 = set(s2.split())
    return len(s1.intersection(s2)) / len(s1.union(s2))


def shingle(string, n_grams=4):
    """
    Shingle a string
    """
    shingles = []
    for i in range(len(string) - n_grams + 1):
        shingles.append(string[i:i+n_grams])
    return shingles


def get_embeddings_simvec(
        items, 
        compare_items, 
        metric=jarowinkler_similarity
        ):
    if compare_items is None:
        compare_items = np.random.choice(items, 128)

    dim = len(compare_items)

    embeddings = cdist(items, compare_items, scorer=metric, workers=1)
    return embeddings


def dedupe_faiss(items, cutoff=0.1, k=10):
    """
    Identify duplicate items 
    """
    compare_items = np.random.choice(items, 128)
    #embeddings = get_embeddings_simvec(items, compare_items, metric=jarowinkler_similarity)

    with mp.Pool(os.cpu_count()) as pool:
        embeddings = pool.starmap(
                get_embeddings_simvec, 
                [(items, compare_items, jarowinkler_similarity)]
                )
    embeddings = np.array(embeddings).squeeze()

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

    match_df['orig_item']  = items[match_df['orig_idxs']]
    match_df['match_item'] = items[match_df['match_idxs']]

    match_df = match_df[match_df['distance'] < cutoff]
    return match_df




if __name__ == '__main__':
    data = pd.read_feather('../data/extracted_addresses.feather')

    data = data[data['address'] != '']
    data = data[data['address'].isnull() == False]

    SUBSET_SIZE = 5_000_000
    random.seed(42)

    data['address'] = data['address'].apply(lambda x: x.strip())
    data = data[data['address'].apply(lambda x: len(x) > 1)]
    addresses = np.random.choice(data['address'], SUBSET_SIZE, replace=False)
    ## Remove nulls
    addresses = addresses[addresses != '']
    addresses = np.unique(addresses)

    print('Number of addresses: {}'.format(len(addresses)))

    fuzzifier = Fuzzifier()

    #query_address = addresses[500]
    query_address = fuzzifier.fuzzify(addresses[500])


    start = perf_counter()
    #match_idxs = get_similar_docs(addresses, query_address, num_perm=8, threshold=0.5)
    match_df = dedupe_faiss(addresses, cutoff=0.1, k=50)

    DISPLAY_COLS = ['orig_item', 'match_item', 'distance']
    print(match_df[DISPLAY_COLS])
    end = perf_counter()

    """
    print(f'\nQuery address: {query_address}')
    print('Matched addresses:')
    print(data.iloc[match_idxs])

    print(f'Actual jaccard similarity: {get_jaccard(query_address, addresses[500])}')
    print('Time taken: {} seconds'.format(end - start))
    """
