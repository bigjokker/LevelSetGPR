# LevelSetGPR: Gaussian Process Toolbox for Function Estimation from Sparse Level Sets

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Overview

LevelSetGPR is a Python toolbox for estimating continuous functions $f: \mathbb{R}^n \to \mathbb{R}$ from sparse, noisy points on a few level sets (e.g., 2–5 levels with 3–4 points each). It uses Gaussian Process Regression (GPR) with custom constraints to incorporate level-set structure, providing probabilistic estimates and uncertainty quantification. Key features include:

- Efficient use of level-set geometry via equality and derivative constraints.
- High-dimensional handling (\( n \leq 100 \)) with optional PCA projection.
- Uncertainty bands: GP (probabilistic), conformal (empirical), Lipschitz (rigorous outer).
- Active sampling for iterative data collection to reduce uncertainty.
- Hyperparameter optimization via marginal likelihood.

Ideal for applications like implicit surface reconstruction, contour estimation, or sparse data interpolation in high dimensions.

# Install dependencies:

pip install numpy scipy


No additional libraries are required for core functionality. For optimization, `scipy.optimize` is used.

## Usage

The toolbox is contained in a single Python script (`levelset_gpr.py`). Import and use the key functions:

- `gp_predict_with_constraints`: Core GPR predictor with level-set constraints.
- `conformal_quantile`: Empirical calibration for finite-sample coverage.
- `robust_L_levels` and `lipschitz_bounds`: For outer Lipschitz envelopes.
- `optimize_hyperparams`: Marginal likelihood optimization for hyperparameters.
- `active_sampling_loop`: Iterative active learning to add points.

### Basic Example

```python
import numpy as np
from levelset_gpr import gp_predict_with_constraints, conformal_quantile, robust_L_levels, lipschitz_bounds

# Toy data: n=10, 2 levels with 3-4 points each
X = np.random.randn(7, 10)  # Points
y = np.array([-0.3, -0.3, 0.9, 0.9, 0.9, -0.3, -0.3])  # Level values
level_groups = [[0,1,5,6], [2,3,4]]  # Indices per level
Xstar = np.random.randn(5, 10)  # Query points

# Fit and predict
mu, var = gp_predict_with_constraints(X, y, level_groups, Xstar, ell=1.0, sf_mat=0.5, sf_lin=1e-4, sf_const=0.4,
                                   sigma_delta=0.01, sigma_deriv=0.01)

# Uncertainty
q_conformal = conformal_quantile(X, y, level_groups, ell=1.0, sf_mat=0.5, sf_lin=1e-4, sf_const=0.4,
                              sigma_delta=0.01, sigma_deriv=0.01)
L = robust_L_levels(X, y, level_groups)
lower, upper = lipschitz_bounds(X, y, Xstar, L)

print("Means:", mu)
print("Vars:", var)
```

# Active Sampling Example:

```python
def oracle_f(x):
    return np.sum(x)  # Replace with real function

def candidates_gen():
    return np.random.uniform(-1, 1, (100, 10))  # Generator: new candidates each iter

new_X, new_y, new_groups, final_params = active_sampling_loop(
    X, y, level_groups,
    candidates_gen_fn=candidates_gen,
    oracle_f=oracle_f,
    num_add=3,
    rule='max_var'
)
```
# Hyperparameter Optimization
Use optimize_hyperparams to tune via marginal likelihood:

```python
init = [np.log(1.0), np.log(0.5), np.log(1e-4), np.log(0.4), np.log(0.01), np.log(0.01)]
bounds = [...]  # As defined in script
opt = optimize_hyperparams(X, y, level_groups, init, bounds)
params = opt['params']
```
