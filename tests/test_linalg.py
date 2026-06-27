import numpy as np
from corenum.linalg import l2_norm

def test_l2_norm():
    """Prove L2 norm matches NumPy's internal linear algebra engine."""
    vector = np.array([3.0, 4.0, 12.0])
    
    # Run corenum function
    our_result = l2_norm(vector)
    
    # Run NumPy's official C-backed function
    numpy_result = np.linalg.norm(vector)
    
    # They are identical up to 7 decimal places
    np.testing.assert_allclose(our_result, numpy_result, rtol=1e-7)

def test_l2_norm_zeros():
    """Edge case: ensure it handles zero vectors correctly."""
    vector = np.zeros(5)
    assert l2_norm(vector) == 0.0