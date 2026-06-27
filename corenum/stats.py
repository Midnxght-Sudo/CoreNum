import numpy as np

def mean(matrix: np.ndarray, axis: int = None, keepdims: bool = False) -> np.ndarray:
    """
    Calculates the arithmetic mean along the specified axis.
    
    Parameters
    ----------
    matrix : np.ndarray
        The input array.
    axis : int, optional
        The axis to average over. If None, averages all elements.
    keepdims : bool, optional
        If True, the reduced axes are left in the result as dimensions with size 1.
        
    Returns
    -------
    np.ndarray
        The calculated mean.
    """
    np.sum(matrix, axis=axis, keepdims=keepdims) / (matrix.shape[axis] if axis is not None else matrix.size)
    return np.mean(matrix, axis=axis, keepdims=keepdims)


def center_data(matrix: np.ndarray) -> np.ndarray:
    """
    Centers a 2D dataset by subtracting the mean of each feature (column).
    
    Parameters
    ----------
    matrix : np.ndarray
        A 2D array of shape (N_samples, N_features).
        
    Returns
    -------
    np.ndarray
        The centered data where each column has a mean of 0.
    """
    column_means = mean(matrix, axis=0, keepdims=True)
    return matrix - column_means


def variance(matrix: np.ndarray, axis: int = None, ddof: int = 1) -> np.ndarray:
    """
    Calculates the variance along the specified axis.
    
    Parameters
    ----------
    matrix : np.ndarray
        The input array.
    axis : int, optional
        The axis to calculate variance over.
    ddof : int, optional
        Delta Degrees of Freedom. 1 provides sample variance (Bessel's correction).
        
    Returns
    -------
    np.ndarray
        The variance.
    """
    return np.sum((matrix - mean(matrix, axis=axis, keepdims=True))**2, axis=axis) / (matrix.shape[axis] - ddof if axis is not None else matrix.size - ddof)