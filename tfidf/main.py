import pandas as pd
import numpy as np
from time import perf_counter
import logging

from tfidf.vector_compression import * 
from utils.address_handler import get_addresses
from utils.fuzzify import Fuzzifier



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    FILENAME = 'data/extracted_addresses.feather'
    data = pd.read_feather(FILENAME)
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

    #match_df = dedupe_knn(addresses)
    #match_df = dedupe_approx_knn(addresses)
    #match_df = dedupe_minibatch_kmeans(addresses)
    #match_df = dedupe_faiss(addresses)

    end = perf_counter()

    print(match_df[match_df['distance'] < 0.08])

    print('Time taken: {} seconds'.format(end - start))
