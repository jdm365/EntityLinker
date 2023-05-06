import pandas as pd

from tfidf.lib.vector_compression import * 
from tfidf.lib.tfidf_standard import *


def test_dedup_faiss():
    FILENAME = 'data/corrupted_companies_dedup.feather'
    data = pd.read_feather(FILENAME)
    data = data.sample(1000)
    dedup_col = 'company'
    dedup_func = dedupe_faiss
    k = 5

    n_duplicates = 0
    for _, value in data['label'].value_counts().items():
        if value > 1:
            n_duplicates += value - 1

    match_df = dedup_func(data[dedup_col], k=k)

    assert match_df.columns.tolist() == ['orig_idxs', 'match_idxs', 'distance'], 'Columns are incorrect'

def test_dedup_knn():
    FILENAME = 'data/corrupted_companies_dedup.feather'
    data = pd.read_feather(FILENAME)
    data = data.sample(1000)
    dedup_col = 'company'
    dedup_func = dedupe_knn
    k = 5

    n_duplicates = 0
    for _, value in data['label'].value_counts().items():
        if value > 1:
            n_duplicates += value - 1

    match_df = dedup_func(data[dedup_col], k=k)

    assert match_df.columns.tolist() == ['orig_idxs', 'match_idxs', 'distance'], 'Columns are incorrect'

def test_dedup_approx_knn():
    FILENAME = 'data/corrupted_companies_dedup.feather'
    data = pd.read_feather(FILENAME)
    data = data.sample(1000)
    dedup_col = 'company'
    dedup_func = dedupe_approx_knn
    k = 5

    n_duplicates = 0
    for _, value in data['label'].value_counts().items():
        if value > 1:
            n_duplicates += value - 1

    match_df = dedup_func(data[dedup_col], k=k)

    assert match_df.columns.tolist() == ['orig_idxs', 'match_idxs', 'distance'], 'Columns are incorrect'
