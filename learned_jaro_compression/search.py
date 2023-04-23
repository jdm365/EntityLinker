import torch as T
import torch.nn.functional as F
from model import train_and_get_embeddings 
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
from time import perf_counter
import faiss
import logging

from utils.fuzzify import Fuzzifier


def encode_string(string, max_len=128):
    encoded_string = []
    for char in string:
        if len(encoded_string) == max_len:
            return T.tensor(encoded_string, dtype=T.float32)
        encoded_string.append(ord(char))

    if len(encoded_string) < max_len:
        encoded_string += [0] * (max_len - len(encoded_string))
    return T.tensor(encoded_string, dtype=T.float32)


def dedupe_faiss(addresses, k=10):
    """
    Remove duplicate addresses
    """
    """
    embeddings = train_and_get_embeddings(
            addresses, 
            n_epochs=0,
            emb_dim=128
            )
    """
    max_str_len = 64
    device = T.device('cuda' if T.cuda.is_available() else 'cpu')
    with T.no_grad():
        rand_proj_matrix = T.randn(max_str_len, 128).to(device)
        embeddings = []
        for address in tqdm(addresses, desc='Encoding addresses'):
            address = encode_string(address, max_len=max_str_len).to(device)
            address = T.matmul(address, rand_proj_matrix)
            embeddings.append(address)
        embeddings = T.stack(embeddings)

        ## Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        embeddings = embeddings.cpu().numpy()

    ## Create index
    dim = embeddings.shape[1]
    quantizer = faiss.IndexFlatL2(dim)

    n_centroids = 8
    n_bits = 8
    n_vornoi_cells = 1024

    index = faiss.IndexIVFPQ(quantizer, dim, n_vornoi_cells, n_centroids, n_bits)
    index.train(embeddings)
    index.add(embeddings)

    index.nprobe = int(np.sqrt(n_vornoi_cells))

    ## Get k nearest neighbours
    distances, indices = index.search(embeddings, k)

    match_df = pd.DataFrame({
        'orig_idxs': np.arange(k * len(addresses)) // k,
        'match_idxs': indices.flatten(),
        'distance': distances.flatten()
    })
    match_df = match_df[match_df['orig_idxs'] != match_df['match_idxs']]
    match_df = match_df[match_df['orig_idxs'] < match_df['match_idxs']]
    match_df = match_df.drop_duplicates(subset=['orig_idxs', 'match_idxs'])

    match_df['orig_address']  = addresses[match_df['orig_idxs']]
    match_df['match_address'] = addresses[match_df['match_idxs']]
    return match_df


    

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    data = pd.read_feather('../data/extracted_addresses.feather')
    data = data[data['address'] != '']
    data = data[data['address'].isnull() == False]

    SUBSET_SIZE = 1_000_000
    random.seed(42)

    data['address'] = data['address'].apply(lambda x: x.strip())
    data['address'] = data['address'].apply(lambda x: x.replace('\t', ''))
    data['address'] = data['address'].apply(lambda x: x.replace('\n', ''))
    data['address'] = data['address'].apply(lambda x: x.replace('  ', ' '))
    data['address'] = data['address'].apply(lambda x: x.lower())
    data['address'] = data['address'].str.strip()
    data = data[data['address'].apply(lambda x: len(x) > 1)]
    addresses = np.random.choice(data['address'], SUBSET_SIZE, replace=False)

    print('Number of addresses: {}'.format(len(addresses)))

    fuzzifier = Fuzzifier()

    #query_address = addresses[500]
    query_address = fuzzifier.fuzzify(addresses[500])

    start = perf_counter()
    match_df = dedupe_faiss(addresses)
    end = perf_counter()

    print(match_df[['orig_address', 'match_address', 'distance']].sort_values('distance', ascending=True).head(25))
    print(match_df[['orig_address', 'match_address', 'distance']].sort_values('distance', ascending=False).head(25))

    print('Time taken: {} seconds'.format(end - start))
