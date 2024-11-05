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

def load_data(filename, yErr):
    data = pd.read_csv(filename, sep='\t', encoding_errors='replace')
    alpha = np.array(data.iloc[:, 0])
    y = np.array(data.iloc[:, 1])
    yErr = np.array(np.full_like(y, yErr))
    # yErr = np.maximum(y*errorPortion, errorMin)

    return alpha, y, yErr

def gauss_fn(x, a, b, c):
    return a * np.exp(-(x-b)**2/ (2*c**2))

def triple_gauss_fn(alpha, a1, b1, c1, a2, b2, c2, a3, b3, c3, B):
    return B + gauss_fn(alpha, a1, b1, c1) + gauss_fn(alpha, a2, b2, c2) + gauss_fn(alpha, a3, b3, c3)

def fit_ccd_data(alpha, y, yErr, p0=None):
    popt, pcov = optimize.curve_fit(triple_gauss_fn, alpha, y, sigma=yErr, p0=p0, maxfev=10000)
    # popt, pcov = optimize.curve_fit(gauss_fn, alpha, y, sigma=yErr, maxfev=10000)
    params, paramsErr = popt, np.sqrt(np.diag(pcov))

    return params, paramsErr

def plot_data_fit(fig, ax, alpha, y, yErr, params):
    # ax.errorbar(alpha, y, yErr, label='Daten')
    ax.plot(alpha, y, '-', lw=1, label='Daten')

    # paramsPrint = np.array((params, paramsErr)).T
    # paramsPrint = (r' \pm '.join(tuple(np.array(param, dtype=str))) for param in paramsPrint)
    # paramsPrint = r',\ '.join(paramsPrint)
    # print(paramsPrint)

    xFit = array_range(alpha, overhang=0)
    yFit = triple_gauss_fn(xFit, *params)
    ax.plot(xFit, yFit, label='Kalibrierungskurve')

# alpha, y, yErr, load_data('p401/data/interference_9.1A.txt', 0.02, 0.5)
# alpha, y, yErr = load_data('p401/data/interference_9.1A.txt', 1)

# fig, ax = plt.subplots()
# plot_data_fit(fig, ax, alpha, y, yErr)
# ax.minorticks_on()
# ax.grid(which='both')
# fig.savefig('p401/plot/gauss_1.pdf')

I = np.array((2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.5, 8.0, 8.5, 9.1))
alphaRange = np.array(((0.77, 1.16), (0.77, 1.16), (0.77, 1.16), (0.77, 1.16), (0.77, 1.16), (0.77, 1.16), (0.77, 1.16), (0.77, 1.16), (0.77, 1.16), (0.77, 1.16)))
inFilenames = tuple('p401/data/interference_%.1fA.txt' % Ival for Ival in I)
outFilenames = tuple('p401/plot/gauss_%.1fA.pdf' % Ival for Ival in I)

params, paramsErr = np.zeros((len(I), 10)), np.zeros((len(I), 10))
for i in range(len(inFilenames)):
# for i in range(9, 10):
    lower, upper = alphaRange[i, 0], alphaRange[i, 1]

    alpha, y, yErr = load_data(inFilenames[i], 1)
    rangeMask = (alpha >= np.full_like(alpha, lower)) & (alpha <= np.full_like(alpha, upper))
    alpha, y, yErr = alpha[rangeMask], y[rangeMask], yErr[rangeMask]

    # fit data
    param, paramErr = fit_ccd_data(alpha, y, yErr, p0=(40, 0.85, 0.05, 50, 0.97, 0.05, 40, 1.09, 0.05, 1))
    params[i] = param
    paramsErr[i] = paramErr

    fig, ax = plt.subplots()

    plot_data_fit(fig, ax, alpha, y, yErr, param)
    ax.legend()
    ax.minorticks_on()
    ax.grid(which='both')
    fig.savefig(outFilenames[i])

print(params)