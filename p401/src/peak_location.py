import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as optimize

def array_range(array, overhang=0.05, nPoints=100):
    min = np.min(array)
    max = np.max(array)
    span = max-min
    min -= overhang*span
    max += overhang*span
    return np.linspace(min, max, nPoints)

def load_data(filename, errorPortion, errorMin):
    data = pd.read_csv(filename, sep='\t')
    alpha = 

    return I, B, Ierr, Berr

def gauss_fn(x, a, b, c):
    return a * np.exp(-(x-b)**2/ (2*c**2))

def triple_gauss_fn(alpha, a1, b1, c1, a2, b2, c2, a3, b3, c3):
    return gauss_fn(alpha, a1, b1, c1) + gauss_fn(alpha, a2, b2, c2) + gauss_fn(alpha, a3, b3, c3)

def fit_ccd_data(alpha, y, yErr):
    popt, pcov = optimize.curve_fit(field_fit_fn, alpha, y, sigma=yErr)
    params, paramsErr = popt, np.sqrt(np.diag(pcov))

    return params, paramsErr

load_data