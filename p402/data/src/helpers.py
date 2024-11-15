import numpy as np
import scipy.optimize as optimize

def array_range(array, overhang=0.05, nPoints=100):
    min = np.min(array)
    max = np.max(array)
    span = max-min
    min -= overhang*span
    max += overhang*span
    return np.linspace(min, max, nPoints)

def linear_fn(x, a, b):
    return a + b*x

def chisq_fit(function, x, y, yErr, p0=None):
    popt, pcov = optimize.curve_fit(function, x, y, sigma=yErr, p0=p0, maxfev=10000, absolute_sigma=True)
    params, paramsErr = popt, np.sqrt(np.diag(pcov))

    return params, paramsErr