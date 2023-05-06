import torch as T 
import pandas as pd
import numpy as np

from char_bert.lib.main import *
from char_bert.lib.utils import *
from char_bert.lib.test_model import HuggingFaceByt5Wrapper, ModelSize


def test_dedup():
    FILENAME = 'data/corrupted_companies_dedup.feather'
    data = pd.read_feather(FILENAME)
    data = data.sample(1000)
    dedup_func = dedupe_byt5
    dedup_col = 'company'
    k = 5 
    model = HuggingFaceByt5Wrapper(model_size=ModelSize.BASE)

    n_duplicates = 0
    for _, value in data['label'].value_counts().items():
        if value > 1:
            n_duplicates += value - 1

    match_df = dedup_func(data[dedup_col], k=k, model=model)

    assert match_df.columns.tolist() == ['orig_idxs', 'match_idxs', 'distance'], 'Columns are incorrect'


def test_sim_search_faiss_byt5_pytest():
    FILENAME = 'data/corrupted_companies_sim_search.feather'
    data = pd.read_feather(FILENAME)
    data = data.sample(1000)
    index_col = 'company_true'
    search_col = 'company_corrupted'
    k = 5

    model = HuggingFaceByt5Wrapper()

    index_embeddings = []
    search_embeddings = []
    for _, batch in enumerate(tqdm(data[index_col].to_list())):
        index_embeddings.append(model.get_embeddings(batch))

    for _, batch in enumerate(tqdm(data[search_col].to_list())):
        search_embeddings.append(model.get_embeddings(batch))
    
    index_embeddings  = T.stack(index_embeddings).cpu().numpy()
    search_embeddings = T.stack(search_embeddings).cpu().numpy()

    ## Create index
    index = create_faiss_index(index_embeddings)

    ## Search index
    distances, indices = index.search(search_embeddings, k=k)

    match_df = pd.DataFrame({
        'orig_idxs': np.arange(k * len(data)) // k,
        'match_idxs': indices.flatten(),
        'distance': distances.flatten()
    })

    match_df['orig_label']  = np.array(data['label'].values)[match_df['orig_idxs'].values.astype(int)]
    match_df['match_label'] = np.array(data['label'].values)[match_df['match_idxs'].values.astype(int)]
    match_df['is_match']    = (match_df['orig_label'] == match_df['match_label']).astype(int)
    
    recall = np.sum(match_df["is_match"]) / len(data)
    assert recall > 0.1, 'Recall too low'
