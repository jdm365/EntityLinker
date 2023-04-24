import pandas as pd
from time import perf_counter

from minhashing.lib.min_hash import *
from minhashing.lib.string_sim_hash import *



if __name__ == '__main__':
    FILENAME = '../data/corrupted_addresses_dedup.feather'
    data = pd.read_feather(FILENAME)

    start = perf_counter()

    match_df = dedupe_faiss(data['address_true'], cutoff=0.1, k=25)

    DISPLAY_COLS = ['orig_item', 'match_item', 'distance']

    end = perf_counter()


    print(match_df[DISPLAY_COLS])

    """
    print(f'\nQuery address: {query_address}')
    print('Matched addresses:')
    print(data.iloc[match_idxs])

    print(f'Actual jaccard similarity: {get_jaccard(query_address, addresses[500])}')
    print('Time taken: {} seconds'.format(end - start))
    """
