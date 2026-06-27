import numpy as np
from .stats import center_data

def l2_norm(vector: np.ndarray) -> float:
    """
    Calculates the L2 (Euclidean) norm of a 1D vector.
    
    Parameters
    ----------
    vector : np.ndarray
        A 1D array.
        
    Returns
    -------
    float
        The L2 norm.
    """
    if vector.ndim != 1:
        raise ValueError("L2 norm is only defined for 1D vectors in this function.")
        
    
    # Square the elements, sum them up, and take the square root using np.sqrt()
    return np.sqrt(np.sum(vector**2))


def pca(matrix: np.ndarray, n_components: int) -> np.ndarray:
    """
    Performs Principal Component Analysis to reduce the dimensionality of the data.
    
    Parameters
    ----------
    matrix : np.ndarray
        A 2D dataset of shape (N_samples, N_features).
    n_components : int
        The number of dimensions to project down to.
        
    Returns
    -------
    np.ndarray
        The compressed dataset of shape (N_samples, n_components).
    """
    N = matrix.shape[0]
      
    centered = center_data((matrix))
    
    cov_matrix = np.dot(centered.T, centered) / (N - 1)
    
   
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    

    top_eigenvectors = sorted_eigenvectors[:, :n_components]
    


    projected_data = np.dot(centered, top_eigenvectors).real
    return projected_data