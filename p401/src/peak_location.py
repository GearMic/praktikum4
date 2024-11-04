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
    alpha = data.iloc[:, 0]
    y = data.iloc[:, 1]
    yErr = np.full_like(y, yErr)
    # yErr = np.maximum(y*errorPortion, errorMin)

    return alpha, y, yErr

def gauss_fn(x, a, b, c):
    return a * np.exp(-(x-b)**2/ (2*c**2))

def triple_gauss_fn(alpha, a1, b1, c1, a2, b2, c2, a3, b3, c3):
    return gauss_fn(alpha, a1, b1, c1) + gauss_fn(alpha, a2, b2, c2) + gauss_fn(alpha, a3, b3, c3)

def fit_ccd_data(alpha, y, yErr):
    popt, pcov = optimize.curve_fit(triple_gauss_fn, alpha, y, sigma=yErr)
    params, paramsErr = popt, np.sqrt(np.diag(pcov))

    return params, paramsErr

def plot_data_fit(fig, ax, alpha, y, yErr):
    # ax.errorbar(alpha, y, yErr, label='Daten')
    ax.plot(alpha, y, '-', lw=1, label='Daten')

    # params, paramsErr = fit_ccd_data(alpha, y, yErr)
    # # paramsPrint = np.array((params, paramsErr)).T
    # # paramsPrint = (r' \pm '.join(tuple(np.array(param, dtype=str))) for param in paramsPrint)
    # # paramsPrint = r',\ '.join(paramsPrint)
    # # print(paramsPrint)
    # print(params)
    # print(paramsErr)

    # xFit = array_range(alpha, overhang=0)
    # yFit = field_fit_fn(xFit, *params)
    # ax.plot(xFit, yFit, label='Kalibrierungskurve')

# alpha, y, yErr, load_data('p401/data/interference_9.1A.txt', 0.02, 0.5)
alpha, y, yErr = load_data('p401/data/interference_9.1A.txt', 1)

fig, ax = plt.subplots()
plot_data_fit(fig, ax, alpha, y, yErr)
ax.minorticks_on()
ax.grid(which='both')
fig.savefig('p401/plot/gauss_1.pdf')

inFilenames = tuple('p401/data/interference_%1fA.txt' % Ival for Ival in I)