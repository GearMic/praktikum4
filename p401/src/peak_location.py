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

def plot_data_fit(fig, ax, alpha, y, yErr, params=None):
    # ax.errorbar(alpha, y, yErr, label='Daten')
    ax.plot(alpha, y, '-', lw=1, label='Daten')

    # paramsPrint = np.array((params, paramsErr)).T
    # paramsPrint = (r' \pm '.join(tuple(np.array(param, dtype=str))) for param in paramsPrint)
    # paramsPrint = r',\ '.join(paramsPrint)
    # print(paramsPrint)

    if not (params is None):
        xFit = array_range(alpha, overhang=0)
        yFit = triple_gauss_fn(xFit, *params)
        ax.plot(xFit, yFit, label='Kalibrierungskurve')

"""
def alpha_to_lambda(alpha, alphaErr):
    alpha *= 2*np.pi/360 # convert to radians
    d = 0.004
    k = 2
    n = 1.457
    lbda = 2*d/k * np.sqrt(n**2-(np.sin(alpha)**2))
    lbdaErr = d/k * (np.sin(2*alpha)**2) / np.sqrt(n**2 - np.sin(alpha)**2) * alphaErr
    return lbda, lbdaErr

def calc_delta_E(lbda1, lbda1, lbda1Err, lbda2Err):
    lbda1, lbda2, lbda3 = lbda
    lbda1Err, lbda2Err, lbda3Err = lbdaErr
    # lbdaDiff1 = np.abs(lbda2 - lbda1)
    # lbdaDiff1Err = np.sqrt(lbda2Err**2 + lbda1Err**2)
    # lbdaDiff2 = np.abs(lbda3 - lbda2)
    # lbdaDiff2Err = np.sqrt(lbda3Err**2 + lbda2Err**2)

    # deltaLbda = (lbdaDiff1 + lbdaDiff2)/2
    # deltaLbdaErr = np.sqrt(lbdaDiff1Err**2 + lbdaDiff2Err**2)/2
"""

def calc_delta_E(lbda1, lbda2, lbda1Err, lbda2Err):
    h = 6.626e-34
    c = 3e8
    lbda0 = 643.8e-9
    deltaE = 0 # TODO

# alpha, y, yErr, load_data('p401/data/interference_9.1A.txt', 0.02, 0.5)

alpha, y, yErr = load_data('p401/data/interference_9.1A.txt', 1)
fig, ax = plt.subplots()
plot_data_fit(fig, ax, alpha, y, yErr)
ax.minorticks_on()
ax.grid(which='both')
fig.savefig('p401/plot/gauss_1.pdf')

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

# mu1 = params[:, 1]
# mu2 = params[:, 2]
# mu3 = params[:, 3]
# mu1Err = paramsErr[:, 1]
# mu2Err = paramsErr[:, 2]
# mu3Err = paramsErr[:, 3]
mu = params[:, (1, 4, 7)].T
muErr = params[:, (1, 4, 7)].T
lbda, lbdaErr = alpha_to_lambda(mu, muErr)
print(lbda)
print(lbdaErr)


fig, ax = plt.subplots()
# ax.plot(I, params[:, 4])
# ax.plot(I, params[:, 7])
# fig.savefig('p401/plot/peak-location.pdf')