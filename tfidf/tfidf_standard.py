import pandas as pd
from tqdm import tqdm
from time import perf_counter
from utils.address_handler import get_addresses
from utils.fuzzify import Fuzzifier
from vector_compression import * 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import faiss
import torch as T
import numpy as np
import sys
from scipy.spatial import KDTree
import logging




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
    return idxs 


def dedupe_faiss(addresses, k=10):
    """
    Remove duplicate addresses
    """
    tfidf = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
    vectors = tfidf.fit_transform(addresses)

    compressed_vectors = compress_vectors(vectors)

    if T.cuda.is_available():
        try:
            resource = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(resource, 0, faiss.IndexFlatL2(compressed_vectors.shape[1]))
        except:
            print('GPU not available')

            m = 8  # number of centroid IDs in final compressed vectors
            bits = 8 # number of bits in each centroid
            nlist = 500  # number of clusters

            ## pad vectors to be a multiple of 8
            compressed_vectors = np.pad(compressed_vectors, ((0, 0), (0, 8 - compressed_vectors.shape[1] % 8)), 'constant')

            quantizer = faiss.IndexFlatL2(compressed_vectors.shape[1])  # we keep the same L2 distance flat index
            index = faiss.IndexIVFPQ(quantizer, compressed_vectors.shape[1], nlist, m, bits) 
            index.train(compressed_vectors)


    index.add(compressed_vectors)
    distances, idxs = index.search(compressed_vectors, k)
    return idxs


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    data = pd.read_feather('data/extracted_addresses.feather')

    SUBSET_SIZE = 100_000

    addresses = np.random.choice(data['address'], SUBSET_SIZE, replace=False)

    fuzzifier = Fuzzifier()

    #query_address = addresses[500]
    query_address = fuzzifier.fuzzify(addresses[500])


    start = perf_counter()
    #match_idxs = get_similar_docs(addresses, query_address)
    match_idxs = dedupe_faiss(addresses)
    end = perf_counter()

    print(f'\nQuery address: {query_address}')
    print('Matched addresses:')
    #print(data.iloc[match_idxs])
    print(match_idxs)

    print('Time taken: {} seconds'.format(end - start))
