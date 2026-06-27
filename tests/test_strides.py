import numpy as np
from corenum.strides import sliding_window_2d

def test_sliding_window_memory():
    """Prove that sliding window does not allocate new memory."""
    matrix = np.arange(1, 17).reshape(4, 4)
    windows = sliding_window_2d(matrix, window_size=3)
    
    assert windows.shape == (2, 2, 3, 3)
    
    assert np.shares_memory(matrix, windows) == True