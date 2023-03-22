from hashlib import sha1
from datasketch import MinHash, MinHashLSH
from utils.address_handler import get_addresses
from time import perf_counter
from tqdm import tqdm



def create_mh_index(data, num_perm=128, threshold=0.6):
    """
    Create MinHash index from a list of data
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm, hashfunc=hash)
    for i, d in enumerate(tqdm(data, desc='Creating MinHash index')):
        m = MinHash(num_perm=num_perm)
        for s in d.split():
            m.update(s.encode('utf8'))
        lsh.insert(i, m)
    return lsh


def get_similar_docs(data, query, num_perm=128, threshold=0.6):
    """
    Get similar documents from a list of data
    """
    lsh = create_mh_index(data, num_perm=num_perm, threshold=threshold)
    m = MinHash(num_perm=num_perm)
    for s in query.split():
        m.update(s.encode('utf8'))
    match_idxs = lsh.query(m)
    return match_idxs 



if __name__ == '__main__':
    data = get_addresses()

    print(data)
    query_address = data[500]


    start = perf_counter()
    match_idxs = get_similar_docs(data, query_address, num_perm=8, threshold=0.8)
    end = perf_counter()
    print(data[match_idxs])
    print('Time taken: {} seconds'.format(end - start))
