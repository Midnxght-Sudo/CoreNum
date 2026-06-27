# CoreNum 

A high-performance numerical computing and linear algebra library built from scratch in Python. 

CoreNum replicates the foundational C-level mechanics of NumPy. It is designed to handle high-dimensional data compression, statistical reductions, and zero-copy memory manipulation with strict computational efficiency.

## Features

* **Zero-Copy Memory Striding:** Implements robust N-dimensional sliding windows without duplicating data in RAM, mirroring raw C pointer logic.
* **Linear Algebra Engine:** Fully vectorized $L_2$ norm calculations and Principal Component Analysis (PCA) for dimensionality reduction.
* **Statistical Reductions:** Vectorized calculations for means, variances (with Bessel's correction), and data centering with strict axis broadcasting.
* **Box-Muller Initialization:** Mathematical transformation of uniform noise into Gaussian (Normal) distributions for neural network weight initialization.
* **Production Infrastructure:** Fully tested via `pytest` with built-in professional logging.

## Installation

Ensure you have Python 3.8+ installed. You can install CoreNum locally in editable mode:

```bash
git clone [https://github.com/YOUR_USERNAME/CoreNum.git](https://github.com/YOUR_USERNAME/CoreNum.git)
cd CoreNum
pip install -e ".[test]"