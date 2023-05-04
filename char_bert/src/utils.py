import torch as T
import pandas as pd
import numpy as np
import faiss

from char_bert.src.main import *


def create_faiss_index(embeddings):
    ## Pad to multiple of 8
    if embeddings.shape[-1] % 8 != 0:
        embeddings = np.pad(
                embeddings, 
                ((0, 0), (0, 8 - embeddings.shape[-1] % 8)),
                'constant', 
                constant_values=0
                )

    index = faiss.index_factory(embeddings.shape[-1], "IVF512,PQ32")
    index.train(embeddings)
    index.add(embeddings)
    index.nprobe = 32

    return index


def create_dedupe_df(idxs, distances):
    deduped_df = pd.DataFrame({
        'orig_idxs': np.arange(len(idxs)),
        'match_idxs': idxs.tolist(), 
        'distance': distances.tolist()
        }).explode(['match_idxs', 'distance'])

    ## Remove self matches
    deduped_df = deduped_df[deduped_df['orig_idxs'] != deduped_df['match_idxs']]
    deduped_df = deduped_df[deduped_df['orig_idxs'] < deduped_df['match_idxs']]
    return deduped_df


def dedupe_char_bert(items, k=5):
    index_embeddings  = get_embeddings(items)

    ## Create index
    index = create_faiss_index(index_embeddings)

    ## Search index
    distances, indices = index.search(index_embeddings, k=k)
    create_dedupe_df(indices, distances)
    return create_dedupe_df(indices, distances)


def dedupe_byt5(items, model, k=5):
    index_embeddings = []
    for batch in tqdm(items.to_list(), desc='Getting Embeddings'):
        index_embeddings.append(model.get_embeddings(batch))

    index_embeddings = T.stack(index_embeddings).cpu().numpy()

    ## Create index
    index = create_faiss_index(index_embeddings)

    ## Search index
    distances, indices = index.search(index_embeddings, k=k)

    create_dedupe_df(indices, distances)
    return create_dedupe_df(indices, distances)

