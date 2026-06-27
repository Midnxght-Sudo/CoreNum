import numpy as np
from corenum.stats import mean, center_data, variance

def test_mean():
    data = np.array([[2.0, 4.0], [4.0, 6.0], [6.0, 8.0]])
    our_mean = mean(data, axis=0)
    np_mean = np.mean(data, axis=0)
    np.testing.assert_allclose(our_mean, np_mean)

def test_variance_ddof():
    """Prove corenum's Bessel's correction matches NumPy."""
    data = np.array([[2.0, 4.0], [4.0, 6.0], [6.0, 8.0]])
    our_var = variance(data, axis=0, ddof=1)
    np_var = np.var(data, axis=0, ddof=1)
    np.testing.assert_allclose(our_var, np_var)