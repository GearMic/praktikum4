import numpy as np
import scipy.optimize as optimize
from scipy.odr import ODR, Model, RealData
import pandas as pd

def plt_preamble():
    pass # TODO: nice font, better looking key box

def read_data_pd(filename, sep=r',\s+'):
    return pd.read_csv(filename, sep=sep, engine='python', comment='#')

def array_range(array, overhang=0.05, nPoints=100):
    min = np.min(array)
    max = np.max(array)
    span = max-min
    min -= overhang*span
    max += overhang*span
    return np.linspace(min, max, nPoints)

def slice_from_range(min, max, x, y):
    mask = (x >= min) & (x <= max)
    return x[mask], y[mask]


def fit_curve(fit_fn, B, x, nPoints=100):
    xFit = array_range(x, nPoints=nPoints)
    yFit = fit_fn(B, xFit)
    return xFit, yFit

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

def format_float_pd_series(df, seriesIndex, nDigitsArr):
    seriesName = df.columns[seriesIndex]
    seriesFormatted = pd.Series(dtype=str)
    # nDigitsArr=np.full_like(np.array(series), 2)
    # # print(nDigitsArr)
    for i in range(len(nDigitsArr)):
        seriesFormatted[i] = f'{df[seriesName][i]:.{nDigitsArr[i]}f}'
    df[seriesName] = seriesFormatted
    

def find_last_zero_digit_index(number):
    return (np.floor(np.log10(np.abs(number))))+1

def format_df(df, columnErrPairs):
    df = df.copy() # otherwise the df would be passed by reference
    nSignificantDigits = 2
    columnsUsed = []

    # format according to the value in the error column
    for i in range(len(columnErrPairs)):
        valIndex, errIndex = columnErrPairs[i]
        if valIndex in columnsUsed or errIndex in columnsUsed:
            raise ValueError('One of the indices %i and %i has already been used' % (valIndex, errIndex)) 
        valColumn = df[df.columns[valIndex]]
        errColumn = df[df.columns[errIndex]]

        nDigitsArr = np.zeros(len(errColumn), dtype=int)
        for j in range(len(errColumn)):
            lastZeroDigit = find_last_zero_digit_index(errColumn[j])
            # if lastZeroDigit > 0:
            #     nDigits = 1
            # else:
            #     nDigits = np.abs(lastZeroDigit)+nSignificantDigits
            nDigits = np.abs(lastZeroDigit)+nSignificantDigits
            # TODO: correctly handle numbers >=1
            nDigitsArr[j] = nDigits

        # df[df.columns[errIndex]] = format_float_pd_series(errColumn, nDigitsArr)
        # df[df.columns[valIndex]] = format_float_pd_series(valColumn, nDigitsArr)
        format_float_pd_series(df, errIndex, nDigitsArr)
        format_float_pd_series(df, valIndex, nDigitsArr)
        columnsUsed.append(valIndex)
        columnsUsed.append(errIndex)
        
    return df

