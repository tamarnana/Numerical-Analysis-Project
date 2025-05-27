import numpy as np
from scipy.optimize import minimize_scalar
from scipy.interpolate import lagrange
from scipy.integrate import simpson

def solve_equation_model(code_churn, test_coverage, defects):
    A = np.vstack((code_churn, test_coverage)).T
    coeffs, _, _, _ = np.linalg.lstsq(A, defects, rcond=None)
    return coeffs[0], coeffs[1]

def find_optimal_testing(test_coverage, defects):
    f = lambda t: np.interp(t, test_coverage, defects)
    res = minimize_scalar(f, bounds=(min(test_coverage), max(test_coverage)), method='bounded')
    return res.x

def interpolate_defects(defects):
    x = np.arange(len(defects))
    poly = lagrange(x, defects)
    mid = len(defects) // 2
    return poly(mid)

def integrate_defects(defects):
    x = np.arange(len(defects))
    return simpson(defects, x)
