import pandas as pd
import numpy as np
from time import perf_counter

from locality_sensitive_hashing.lib.min_hash import *
from locality_sensitive_hashing.lib.string_sim_hash import *




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

    match_df['orig_label']  = np.array(data['label'].values)[match_df['orig_idxs'].values]
    match_df['match_label'] = np.array(data['label'].values)[match_df['match_idxs'].values]

    match_df['is_match'] = (match_df['orig_label'] == match_df['match_label']).astype(int)

    print(f'Accuracy:  {np.sum(match_df["is_match"]) / len(match_df)}')
    print(f'Recall:    {np.sum(match_df["is_match"]) / n_duplicates}')

    print('Time taken: {} seconds'.format(perf_counter() - start))


def test_sim_search_faiss(data, index_col, search_col, cutoff=0.08, k=10):
    """
    Test similarity search function
    @param sim_search_func: similarity search function
    @param data: dataframe containing data to search
    @param search_col: column to search
    @param kwargs: keyword arguments to pass to similarity search function
    return: None
    """
    start = perf_counter()

    items = np.array(data[index_col])
    compare_items = np.random.choice(items, 128)
    embeddings = get_embeddings_simvec(items, compare_items, metric=jarowinkler_similarity)

    """
    with mp.Pool(os.cpu_count()) as pool:
        embeddings = pool.starmap(
                get_embeddings_simvec, 
                [(items, compare_items, jarowinkler_similarity)]
                )
    """
    embeddings = np.array(embeddings).squeeze()

    ## Normalize embeddings
    #embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, None]

    ## Create index
    index = create_faiss_index(embeddings)

    embeddings = get_embeddings_simvec(data[search_col], compare_items, metric=jarowinkler_similarity)

    ## Search index
    distances, indices = index.search(embeddings, k)

    match_df = pd.DataFrame({
        'orig_idxs': np.arange(k * len(items)) // k,
        'match_idxs': indices.flatten(),
        'distance': distances.flatten()
    })

    match_df = match_df[match_df['distance'] < cutoff]

    match_df['orig_label']  = np.array(data['label'].values)[match_df['orig_idxs'].values]
    match_df['match_label'] = np.array(data['label'].values)[match_df['match_idxs'].values]
    match_df['is_match']    = (match_df['orig_label'] == match_df['match_label']).astype(int)

    print(f'Accuracy:  {np.sum(match_df["is_match"]) / len(match_df)}')
    print(f'Recall:    {np.sum(match_df["is_match"]) / len(data)}')

    print('Time taken: {} seconds'.format(perf_counter() - start))


def test_dedup_lsh(data, dedup_col, num_perm=128, threshold=0.6):
    """
    Test deduplication function
    @param data: dataframe containing data to deduplicate
    @param dedup_col: column to deduplicate
    @param num_perm: number of permutations to use for minhashing
    @param threshold: threshold for LSH
    return: None
    """
    n_duplicates = 0
    for _, value in data['label'].value_counts().items():
        if value > 1:
            n_duplicates += value - 1

    start = perf_counter()
    duplicate_idxs = dedupe_lsh(data[dedup_col].values, num_perm=num_perm, threshold=threshold)
    orig_idxs = np.arange(len(data))

    match_df = pd.DataFrame({
        'orig_idxs':  orig_idxs,
        'match_idxs': duplicate_idxs
    }).explode('match_idxs')

    match_df = match_df[match_df['orig_idxs'] != match_df['match_idxs']]
    match_df['orig_label']  = np.array(data['label'].values)[match_df['orig_idxs'].values.astype(int)]
    match_df['match_label'] = np.array(data['label'].values)[match_df['match_idxs'].values.astype(int)]
    match_df['is_match'] = (match_df['orig_label'] == match_df['match_label']).astype(int)

    print(f'Accuracy:  {np.sum(match_df["is_match"]) / len(match_df)}')
    print(f'Recall:    {np.sum(match_df["is_match"]) / n_duplicates}')

    print('Time taken: {} seconds'.format(perf_counter() - start))


def test_sim_search_minhash(data, index_col, search_col, num_perm=128, threshold=0.6):
    """
    Test similarity search function
    @param data: dataframe containing data to search
    @param index_col: column to use for index
    @param search_col: column to search
    @param num_perm: number of permutations to use for minhashing
    @param threshold: threshold for LSH
    return: None
    """
    start = perf_counter()

    ## Create index
    index = create_mh_index(data[index_col].values, num_perm=num_perm, threshold=threshold)

    ## Search index
    match_idxs = []
    for _, string in enumerate(tqdm(data[search_col].values, desc='Getting Matches')):
        minhash = hash_string(string, num_perm=num_perm)
        match_idxs.append(index.query(minhash))

    match_df = pd.DataFrame({
        'orig_idxs': np.arange(len(data)),
        'match_idxs': match_idxs,
    }).explode('match_idxs')

    match_df = match_df.dropna(subset=['match_idxs'])
    print(match_df)
    match_df['orig_label']  = np.array(data['label'].values)[match_df['orig_idxs'].values.astype(int)]
    match_df['match_label'] = np.array(data['label'].values)[match_df['match_idxs'].values.astype(int)]
    match_df['is_match']    = (match_df['orig_label'] == match_df['match_label']).astype(int)

    print(f'Accuracy:  {np.sum(match_df["is_match"]) / len(match_df)}')
    print(f'Recall:    {np.sum(match_df["is_match"]) / len(data)}')

    print('Time taken: {} seconds'.format(perf_counter() - start))






if __name__ == '__main__':
    #FILENAME = '../data/corrupted_addresses_dedup.feather'
    #FILENAME = '../data/corrupted_companies_dedup.feather'
    #data = pd.read_feather(FILENAME)

    #test_dedup(dedupe_faiss, data, 'address_true', cutoff=0.08, k=25)
    #test_dedup(dedupe_faiss, data, 'company', cutoff=0.08, k=25)

    #FILENAME = '../data/corrupted_addresses_dedup.feather'
    FILENAME = '../data/corrupted_companies_sim_search.feather'
    data = pd.read_feather(FILENAME)

    #test_sim_search_faiss(data, 'address_true', 'address_corrupted', cutoff=0.08, k=25)
    #test_sim_search_faiss(data, 'company_true', 'company_corrupted', cutoff=0.30, k=10)

    #test_dedup_lsh(data, 'company', threshold=0.8, num_perm=128)
    test_sim_search_minhash(data, 'company_true', 'company_corrupted', threshold=0.5, num_perm=256)
