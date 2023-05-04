import pandas as pd
import numpy as np
from time import perf_counter
from sklearn.neighbors import NearestNeighbors
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


def test_sim_search_knn(data, index_col, search_col, k=10):
    """
    Test similarity search function
    @param sim_search_func: similarity search function
    @param data: dataframe containing data to search
    @param search_col: column to search
    @param kwargs: keyword arguments to pass to similarity search function
    return: None
    """
    start = perf_counter()

    index_embeddings, tfidf = get_embeddings(data[index_col].values)
    search_embeddings       = tfidf.transform(data[search_col].values)

    ## Create index
    neighbors = NearestNeighbors(n_neighbors=k, n_jobs=-1, algorithm='kd_tree')
    neighbors.fit(index_embeddings)

    ## Search index
    distances, indices = neighbors.kneighbors(search_embeddings)

    match_df = pd.DataFrame({
        'orig_idxs': np.arange(len(data)),
        'match_idxs': indices.tolist(),
    }).explode('match_idxs')

    match_df['orig_label']  = np.array(data['label'].values)[match_df['orig_idxs'].values.astype(int)]
    match_df['match_label'] = np.array(data['label'].values)[match_df['match_idxs'].values.astype(int)]
    match_df['is_match']    = (match_df['orig_label'] == match_df['match_label']).astype(int)

    print(f'Recall:    {np.sum(match_df["is_match"]) / len(data)}')

    print('Time taken: {} seconds'.format(perf_counter() - start))


def test_sim_search_approx_knn(data, index_col, search_col, k=10):
    """
    Test similarity search function
    @param sim_search_func: similarity search function
    @param data: dataframe containing data to search
    @param search_col: column to search
    @param kwargs: keyword arguments to pass to similarity search function
    return: None
    """
    start = perf_counter()

    index_embeddings, tfidf = get_embeddings(data[index_col].values)
    search_embeddings       = tfidf.transform(data[search_col].values)

    ## Create index
    index = get_approx_knn_index(index_embeddings, k=k)

    ## Search index
    indices, distances = index.query(search_embeddings, k=k)

    match_df = pd.DataFrame({
        'orig_idxs': np.arange(k * len(data)) // k,
        'match_idxs': indices.flatten(),
        'distance': distances.flatten()
    })

    match_df['orig_label']  = np.array(data['label'].values)[match_df['orig_idxs'].values.astype(int)]
    match_df['match_label'] = np.array(data['label'].values)[match_df['match_idxs'].values.astype(int)]
    match_df['is_match']    = (match_df['orig_label'] == match_df['match_label']).astype(int)

    print(f'Accuracy:  {np.sum(match_df["is_match"]) / len(match_df)}')
    print(f'Recall:    {np.sum(match_df["is_match"]) / len(data)}')

    print('Time taken: {} seconds'.format(perf_counter() - start))


def test_sim_search_faiss(data, index_col, search_col, k=10):
    """
    Test similarity search function
    @param data: dataframe containing data to search
    @index_col: column to index
    @param search_col: column to search
    @param kwargs: keyword arguments to pass to similarity search function
    return: None
    """
    start = perf_counter()

    index_embeddings, tfidf = get_compressed_embeddings(data[index_col].values, dim=64, return_tfidf=True)
    search_embeddings       = compress_vectors(tfidf.transform(data[search_col].values), n_singular_values=64)

    ## Create index
    index = create_faiss_index(index_embeddings)

    ## Search index
    distances, indices = index.search(search_embeddings, k=k)

    match_df = pd.DataFrame({
        'orig_idxs': np.arange(len(data)),
        'match_idxs': indices.tolist(),
        'distance': distances.tolist()
    }).explode(['match_idxs', 'distance'])

    match_df['orig_label']  = np.array(data['label'].values)[match_df['orig_idxs'].values.astype(int)]
    match_df['match_label'] = np.array(data['label'].values)[match_df['match_idxs'].values.astype(int)]
    match_df['is_match']    = (match_df['orig_label'] == match_df['match_label']).astype(int)

    print(f'Accuracy:  {np.sum(match_df["is_match"]) / len(match_df)}')
    print(f'Recall:    {np.sum(match_df["is_match"]) / len(data)}')

    print('Time taken: {} seconds'.format(perf_counter() - start))




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    #FILENAME = '../data/corrupted_addresses_dedup.feather'
    FILENAME = 'data/corrupted_companies_dedup.feather'
    data = pd.read_feather(FILENAME)

    #test_dedup(dedupe_faiss, data, 'address_true', k=5)
    #test_dedup(dedupe_faiss, data, 'company', k=5)
    #test_dedup(dedupe_knn, data, 'company', k=5)
    #test_dedup(dedupe_approx_knn, data, 'company', k=5)
    test_dedup(dedupe_bm25, data, 'company', k=5)

    #FILENAME = '../data/corrupted_addresses_dedup.feather'
    #FILENAME = '../data/corrupted_companies_sim_search.feather'
    #data = pd.read_feather(FILENAME)

    #test_sim_search_faiss(data, 'address_true', 'address_corrupted', cutoff=0.08, k=25)
    #test_sim_search_knn(data, 'company_true', 'company_corrupted', k=5)
    #test_sim_search_approx_knn(data, 'company_true', 'company_corrupted', k=5)
