import pandas as pd
import numpy as np

from locality_sensitive_hashing.lib.min_hash import *
from locality_sensitive_hashing.lib.string_sim_hash import *




def test_dedup_lsh():
    FILENAME = 'data/corrupted_companies_dedup.feather'
    data = pd.read_feather(FILENAME)
    data = data.sample(1000)
    dedup_col = 'company'
    num_perm = 128
    threshold = 0.6

    n_duplicates = 0
    for _, value in data['label'].value_counts().items():
        if value > 1:
            n_duplicates += value - 1

    duplicate_idxs = dedupe_lsh(data[dedup_col].values, num_perm=num_perm, threshold=threshold)
    orig_idxs = np.arange(len(data))

    match_df = pd.DataFrame({
        'orig_idxs':  orig_idxs,
        'match_idxs': duplicate_idxs
    }).explode('match_idxs')


    assert match_df.columns.tolist() == ['orig_idxs', 'match_idxs'], 'Columns are incorrect'


def test_sim_search_minhash():
    FILENAME = 'data/corrupted_companies_sim_search.feather'
    data = pd.read_feather(FILENAME)
    data = data.sample(1000)
    index_col = 'company_true'
    search_col = 'company_corrupted'
    num_perm = 128
    threshold = 0.6

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
    match_df = match_df.fillna(-1)

    match_df['orig_label']  = np.array(data['label'].values)[match_df['orig_idxs'].values.astype(int)]
    match_df['match_label'] = np.array(data['label'].values)[match_df['match_idxs'].values.astype(int)]
    match_df['is_match']    = (match_df['orig_label'] == match_df['match_label']).astype(int)

    recall = np.sum(match_df["is_match"]) / len(data)
    assert recall > 0.1, 'Recall is too low'
