from model import train_and_get_embeddings 
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from time import perf_counter
import faiss
import torch as T
import logging
import sys

sys.path.append('../')
from utils.address_handler import get_addresses
from utils.fuzzify import Fuzzifier




def dedupe_faiss(addresses, k=10):
    """
    Remove duplicate addresses
    """
    embeddings = train_and_get_embeddings(
            addresses, 
            n_epochs=1,
            emb_dim=128
            )

    ## Create index
    dim = embeddings.shape[1]
    quantizer = faiss.IndexFlatL2(dim)

    n_centroids = 8
    n_bits = 8
    n_vornoi_cells = 1024

    index = faiss.IndexIVFPQ(quantizer, dim, n_vornoi_cells, n_centroids, n_bits)
    index.nprobe = int(np.sqrt(n_vornoi_cells))
    index.train(embeddings)
    index.add(embeddings)

    ## Get k nearest neighbours
    distances, indices = index.search(embeddings, k)

    match_df = pd.DataFrame({
        'orig_idxs': np.arange(k * len(addresses)) // k,
        'match_idxs': indices.flatten(),
        'distance': distances.flatten()
    })
    match_df = match_df[match_df['orig_idxs'] != match_df['match_idxs']]
    match_df = match_df.drop_duplicates(subset=['orig_idxs', 'match_idxs'])

    match_df['orig_address']  = addresses[match_df['orig_idxs']]
    match_df['match_address'] = addresses[match_df['match_idxs']]
    return match_df


    

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    data = pd.read_feather('../data/extracted_addresses.feather')
    data = data[data['address'] != '']
    data = data[data['address'].isnull() == False]

    SUBSET_SIZE = 500_000
    random.seed(42)

    data['address'] = data['address'].apply(lambda x: x.strip())
    data = data[data['address'].apply(lambda x: len(x) > 1)]
    addresses = np.random.choice(data['address'], SUBSET_SIZE, replace=False)
    ## Remove nulls
    addresses = addresses[addresses != '']


    print('Number of addresses: {}'.format(len(addresses)))

    fuzzifier = Fuzzifier()

    #query_address = addresses[500]
    query_address = fuzzifier.fuzzify(addresses[500])


    start = perf_counter()
    match_df = dedupe_faiss(addresses)
    end = perf_counter()

    print(match_df[['orig_address', 'match_address', 'distance']].sort_values('distance', ascending=True).head(25))

    print('Time taken: {} seconds'.format(end - start))
