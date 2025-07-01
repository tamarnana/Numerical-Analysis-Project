import numpy as np
from scipy.optimize import minimize_scalar
from scipy.interpolate import lagrange
from scipy.integrate import simpson

def solve_equation_model(code_churn, test_coverage, defects):
    """
    Solve a linear regression model defects = a * code_churn + b * test_coverage.

    Parameters:
        code_churn (array-like): independent variable 1
        test_coverage (array-like): independent variable 2
        defects (array-like): dependent variable

    Returns:
        tuple: (a, b) coefficients of the linear model
    """
    A = np.vstack((code_churn, test_coverage)).T  # shape (n_samples, 2)
    coeffs, _, _, _ = np.linalg.lstsq(A, defects, rcond=None)
    return coeffs[0], coeffs[1]

def find_optimal_testing(test_coverage, defects):
    """
    Find the testing coverage value that minimizes defects using scalar minimization.

    Parameters:
        test_coverage (array-like): x-values for defects
        defects (array-like): y-values for defects

    Returns:
        float: optimal test coverage value minimizing defects
    """
    f = lambda t: np.interp(t, test_coverage, defects)  # interpolation function
    res = minimize_scalar(f, bounds=(min(test_coverage), max(test_coverage)), method='bounded')
    return res.x

def interpolate_defects(defects):
    """
    Interpolate defects using Lagrange polynomial and evaluate at midpoint index.

    Parameters:
        defects (array-like): defect values

    Returns:
        float: interpolated defect value at midpoint
    """
    x = np.arange(len(defects))
    poly = lagrange(x, defects)
    mid = len(defects) // 2
    return poly(mid)

def integrate_defects(defects):
    """
    Compute numerical integration of defects using Simpson's rule.

    Parameters:
        defects (array-like): defect values

    Returns:
        float: definite integral over defects indices
    """
    x = np.arange(len(defects))
    return simpson(defects, x)
