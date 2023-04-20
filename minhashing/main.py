import pandas as pd
import pandas as pd
import numpy as np
import random
from time import perf_counter
#from hashlib import sha1

from minhashing.min_hash import *
from minhashing.string_sim_hash import *
from utils.address_handler import get_addresses
from utils.fuzzify import Fuzzifier



if __name__ == '__main__':
    FILENAME = 'data/extracted_addresses.feather'
    data = pd.read_feather(FILENAME)

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
