import numpy as np
from scipy.sparse.linalg import svds
import logging




def compress_vectors(vectors, n_singluar_values=None):
    """
    Compress vectors using SVD for similarity search.
    -----------------------------------------------------------------------------------------
    Basic Idea is we want to calculate similarity matrix XX^T where X=vectors.
    Dimensionality of vectors is very high and doing sparse matrix multiplication 
    or nearest neighbor search is very slow and memory intensive.
    We can reduce the dimensionality of vectors by using SVD and then do nearest neighbor search.

    SVD tells us we can decompose a matrix X into 3 matrices U, S, V^T such that
    X = USV^T
    therefore
    X^T = VS^TU^T

    U is also orthonormal matrix, so U^TU = I
    V is also orthonormal matrix, so V^TV = I
    
    so

    XX^T = USV^T VS^TU^T = US^2U^T

    Therefore we can represent the vectors with U @ S where U @ S is a matrix of dimension N x k
    We can decrease k by calculating only the top k singular values and vectors to get an
    approximation of U @ S.

    Read more here -> https://web.mit.edu/be.400/www/SVD/Singular_Value_Decomposition.htm
    -----------------------------------------------------------------------------------------


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
