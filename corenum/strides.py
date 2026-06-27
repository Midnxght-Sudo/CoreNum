import numpy as np
from numpy.lib.stride_tricks import as_strided

def sliding_window_2d(matrix: np.ndarray, window_size: int) -> np.ndarray:
    """
    Extracts overlapping patches from a 2D matrix without duplicating memory.
    
    Parameters
    ----------
    matrix : np.ndarray
        The input 2D array.
    window_size : int
        The size of the square window (e.g., 3 for a 3x3 window).
        
    Returns
    -------
    np.ndarray
        A 4D array of shape (out_rows, out_cols, window_size, window_size)
        containing the overlapping views.
        
    Raises
    ------
    ValueError
        If the input matrix is not 2D or if the window size is larger than the matrix.
    """
    # 1. Error Handling
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got {matrix.ndim}D.")
    
    rows, cols = matrix.shape
    if window_size > rows or window_size > cols:
        raise ValueError("Window size cannot be larger than the matrix dimensions.")
        
    # 2. Calculate the shape of the new 4D view
    out_rows = rows - window_size + 1
    out_cols = cols - window_size + 1
    
    new_shape = (out_rows, out_cols, window_size, window_size)
    
    # 3. Calculate the strides
    row_stride, col_stride = matrix.strides
    
    new_strides = (row_stride, col_stride, row_stride, col_stride)
    
    # 4. Generate and return the view
    return as_strided(matrix, shape=new_shape, strides=new_strides)