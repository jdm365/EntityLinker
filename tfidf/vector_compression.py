import numpy as np
from scipy.sparse.linalg import svds
import logging




def compress_vectors(vectors, n_singluar_values=None):
    """
    Compress vectors using SVD
    vectors - N vectors of dimension D
    compressed_vectors - N vectors of dimension k
    k << D
    """
    if n_singluar_values is None:
        n_singluar_values = int(np.sqrt(vectors.shape[1]))
    u, s, vt = svds(vectors, k=n_singluar_values)
    compressed_vectors = (u @ np.diag(s)).astype(np.float32)
    logging.info(f'Compressed vectors from {vectors.shape[1]} to {compressed_vectors.shape[1]}')
    return compressed_vectors

