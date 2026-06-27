import numpy as np

def uniform(size: tuple) -> np.ndarray:
    """
    Generates uniform random numbers between 0.0 and 1.0.
    """
    return np.random.uniform(0.0, 1.0, size=size)

def normal(size: tuple) -> np.ndarray:
    """
    Generates Gaussian (normal) numbers using the Box-Muller transform.
    Converts uniform noise into a standard normal distribution (mean=0, variance=1).
    """
    u1 = uniform(size)
    u2 = uniform(size)
    
    u1 = np.clip(u1, 1e-15, 1.0)
    
    # TODO 1: Implement the Box-Muller formula mathematically.
    # Z = sqrt(-2 * ln(u1)) * cos(2 * pi * u2)
    # Hint: Use np.sqrt(), np.log(), np.cos(), and np.pi
    z = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    return z