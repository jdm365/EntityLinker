from datasketch import MinHash, MinHashLSH
from tqdm import tqdm



def create_mh_index(data, num_perm=128, threshold=0.6):
    """
    Create MinHash index from a list of data
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm, hashfunc=hash)
    for i, d in enumerate(tqdm(data, desc='Creating MinHash index')):
        m = MinHash(num_perm=num_perm)
        for s in shingle(d):
            m.update(s.encode('utf8'))
        lsh.insert(i, m)
    return lsh


def get_similar_docs(data, query, num_perm=128, threshold=0.6):
    """
    Get similar documents from a list of data
    """
    lsh = create_mh_index(data, num_perm=num_perm, threshold=threshold)
    m = MinHash(num_perm=num_perm)
    for s in shingle(query):
        m.update(s.encode('utf8'))
    match_idxs = lsh.query(m)
    return match_idxs 


def get_jaccard(s1, s2):
    """
    Get Jaccard similarity between two strings
    """
    s1 = set(s1.split())
    s2 = set(s2.split())
    return len(s1.intersection(s2)) / len(s1.union(s2))


def shingle(string, n_grams=4):
    """
    Shingle a string
    """
    shingles = []
    for i in range(len(string) - n_grams + 1):
        shingles.append(string[i:i+n_grams])
    return shingles
