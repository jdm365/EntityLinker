import pandas as pd
import numpy as np
from time import perf_counter
import logging

from tfidf.lib.vector_compression import * 
from tfidf.lib.tfidf_standard import *


def test_dedup(dedup_func, data, dedup_col, **kwargs):
    """
    Test deduplication function
    @param dedup_func: deduplication function
    @param data: dataframe containing data to deduplicate
    @param dedup_col: column to deduplicate
    @param kwargs: keyword arguments to pass to deduplication function
    return: None
    """
    n_duplicates = 0
    for _, value in data['label'].value_counts().items():
        if value > 1:
            n_duplicates += value - 1

    start = perf_counter()
    match_df = dedup_func(data[dedup_col], **kwargs)

    match_df['orig_label']  = np.array(data['label'].values)[match_df['orig_idxs'].values.astype(int)]
    match_df['match_label'] = np.array(data['label'].values)[match_df['match_idxs'].values.astype(int)]

    match_df['is_match'] = (match_df['orig_label'] == match_df['match_label']).astype(int)

    print(f'Accuracy:  {np.sum(match_df["is_match"]) / len(match_df)}')
    print(f'Recall:    {np.sum(match_df["is_match"]) / n_duplicates}')

    print('Time taken: {} seconds'.format(perf_counter() - start))



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    FILENAME = '../data/corrupted_companies_dedup.feather'
    data = pd.read_feather(FILENAME)

    #test_dedup(dedupe_faiss, data, 'company', k=5)
    #test_dedup(dedupe_knn, data, 'company', k=5)
    test_dedup(dedupe_approx_knn, data, 'company', k=5)

    #FILENAME = '../data/corrupted_companies_sim_search.feather'
    #data = pd.read_feather(FILENAME)

    #test_sim_search_knn(data, 'company_true', 'company_corrupted', k=5)
    #test_sim_search_approx_knn(data, 'company_true', 'company_corrupted', k=5)
