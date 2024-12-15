import numpy as np
import scipy.optimize as optimize
from scipy.odr import ODR, Model, RealData
import pandas as pd

def plt_preamble():
    pass # TODO: nice font, better looking key box

def read_data_pd(filename):
    return pd.read_csv(filename, sep=r',\s+', engine='python', comment='#')

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

def linear_fn_odr(B, x):
    a, b = B
    return a + b*x

def chisq_fit(function, x, y, yErr=None, p0=None, bounds=None, absolute_sigma=False):
    popt, pcov = optimize.curve_fit(function, x, y, sigma=yErr, p0=p0, maxfev=10000, absolute_sigma=absolute_sigma, bounds=(-np.inf, np.inf))
    params, paramsErr = popt, np.sqrt(np.diag(pcov))

    return params, paramsErr

def odr_fit(function, x, y, nParams, xErr=None, yErr=None, p0=None):
    if p0 is None:
        p0 = np.ones(nParams)

    data = RealData(x, y, sx=xErr, sy=yErr)
    model = Model(function)
    odr = ODR(data, model, beta0=p0)
    output = odr.run()
    return output.beta, output.sd_beta

def format_pd_series(series, nDigitsArr):
    for i in range(len(nDigitsArr)):
        series[i] = f'{valColumn[i]:.{nDigitsArr[i]}f}'
    return series

def format_df(dataFrame, columnErrPairs):
    nSignificantDigits = 2

    for i in range(len(errPairs)):
        valIndex, errIndex = errorPairs[i]
        valColumn = dataFrame[dataFrame.columns[valIndex]]
        errColumn = dataFrame[dataFrame.columns[errIndex]]

        nDigitsArr = np.zeros(len(errColumn))
        for j in range(len(errColumn)):
            lastZeroDigit = -(np.floor(np.log10(np.abs(errColumn[j])))-1)
            # if lastZeroDigit < 0:
            #     nDigits = 1
            # else:
            #     nDigits = firstNonzeroDigit+1
            nDigits = lastZeroDigit+nSignificantDigits

        dataFrame[dataFrame.columns[errIndex]] = format_pd_series(errColumn, nDigitsArr)
        dataFrame[dataFrame.columns[valIndex]] = format_pd_series(valColumn, nDigitsArr)
        
    return dataFrame

