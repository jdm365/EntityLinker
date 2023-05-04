from datasketch import MinHash, MinHashLSH
from tqdm import tqdm


def get_jaccard(s1, s2):
    """
    Get Jaccard similarity between two strings
    @param s1: string 1
    @param s2: string 2
    @return: Jaccard similarity
    """
    s1 = set(s1.split())
    s2 = set(s2.split())
    return len(s1.intersection(s2)) / len(s1.union(s2))

def shingle(string, n_grams=4):
    """
    Shingle a string
    @param string: string to shingle
    @param n_grams: number of n-grams to use
    @return: list of n-grams
    """
    shingles = []
    for idx in range(len(string) - n_grams + 1):
        shingles.append(string[idx:idx+n_grams])
    return shingles

def create_mh_index(data, num_perm=128, threshold=0.6):
    """
    Create MinHash index from a list of data
    @param data: list of strings
    @param num_perm: number of permutations to use
    @param threshold: Jaccard similarity threshold
    @return: MinHashLSH index
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm, hashfunc=hash)
    for i, d in enumerate(tqdm(data, desc='Creating MinHash index')):
        m = MinHash(num_perm=num_perm)
        for s in shingle(d):
            m.update(s.encode('utf8'))
        lsh.insert(i, m)
    return lsh

def hash_string(string, num_perm=128):
    """
    Hash a string using MinHash
    @param string: string to hash
    @param num_perm: number of permutations to use
    @return: MinHash object
    """
    minhash = MinHash(num_perm=num_perm)
    for shngl in shingle(string):
        minhash.update(shngl.encode('utf8'))
    return minhash


def dedupe_lsh(data, num_perm=128, threshold=0.6):
    """
    Deduplicate a list of strings using MinHash LSH
    @param data: list of strings
    @param num_perm: number of permutations to use
    @param threshold: Jaccard similarity threshold
    @return: list of duplicate idxs
    """
    # Create MinHash LSH index
    lsh = MinHashLSH(threshold=threshold, num_perm=128)

    # Insert the strings into the index with their MinHash
    for idx, string in enumerate(tqdm(data, desc='Creating MinHash index')):
        minhash = hash_string(string, num_perm=num_perm)
        lsh.insert(idx, minhash)

    # Deduplicate the strings and report duplicate idxs
    unique_idxs = []
    for idx, string in enumerate(tqdm(data, desc='Deduplicating strings')):
        minhash = hash_string(string, num_perm=num_perm)
        dups = lsh.query(minhash)
        unique_idxs.append(dups)

    return unique_idxs
