import numpy as np
import scipy.optimize as optimize

def array_range(array, overhang=0.05, nPoints=100):
    min = np.min(array)
    max = np.max(array)
    span = max-min
    min -= overhang*span
    max += overhang*span
    return np.linspace(min, max, nPoints)

def mean_mean_err(value1, value2, value1Err, value2Err):
    mean = (value1 + value2) / 2
    meanErr = np.sqrt(value1Err**2 + value2Err**2) / 2
    return mean, meanErr

def linear_fn(x, a, b):
    return a + b*x

def chisq_fit(function, x, y, yErr, p0=None, bounds=None):
    popt, pcov = optimize.curve_fit(function, x, y, sigma=yErr, p0=p0, maxfev=10000, absolute_sigma=True, bounds=(-np.inf, np.inf))
    params, paramsErr = popt, np.sqrt(np.diag(pcov))

    return params, paramsErr