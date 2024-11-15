import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as optimize
from helpers import *

def load_data(filename, yErr):
    data = pd.read_csv(filename, sep='\t', encoding_errors='replace', decimal=',')
    # alpha = np.array(data.iloc[:, 0], dtype=float)
    # y = np.array(data.iloc[:, 1], dtype=float)
    # yErr = np.array(np.full_like(y, yErr), dtype=float)
    alpha = np.array(data.iloc[:, 0])
    y = np.array(data.iloc[:, 1])
    yErr = np.array(np.full_like(y, yErr))
    # yErr = np.maximum(y*errorPortion, errorMin)
    # print(alpha.dtype, y.dtype, yErr.dtype)

    return alpha, y, yErr

def gauss_fn(x, a, b, c):
    return a * np.exp(-(x-b)**2/ (2*c**2))

def double_gauss_fn(alpha, B, a1, mu1, sigma1, a2, mu2, sigma2):
    return B + gauss_fn(alpha, a1, mu1, sigma1) + gauss_fn(alpha, a2, mu2, sigma2)

def plot_data_fit(fig, ax, alpha, y, yErr, params=None):
    ax.errorbar(alpha, y, yErr, fmt='x', label='Daten')
    # ax.plot(alpha, y, '-', lw=1, label='Daten')

    # paramsPrint = np.array((params, paramsErr)).T
    # paramsPrint = (r' \pm '.join(tuple(np.array(param, dtype=str))) for param in paramsPrint)
    # paramsPrint = r',\ '.join(paramsPrint)
    # print(paramsPrint)

    if not (params is None):
        xFit = array_range(alpha, overhang=0)
        yFit = double_gauss_fn(xFit, *params)
        ax.plot(xFit, yFit, label='Kalibrierungskurve')


omegaG = np.array((13.8, 18.2, 37.0))
alphaRange = np.array(((0.77, 1.16), (0.77, 1.16), (0.77, 1.16)))
inFilenames = tuple('p402/data/ccd/line%.1f.txt' % omega for omega in omegaG)
outFilenames = tuple('p401/plot/gauss%.1fA.pdf' % omega for omega in omegaG)

params, paramsErr = np.zeros((len(omegaG), 7)), np.zeros((len(omegaG), 7))
for i in range(len(inFilenames)):
# for i in range(9, 10):
    lower, upper = alphaRange[i, 0], alphaRange[i, 1]

    alpha, y, yErr = load_data(inFilenames[i], 1)
    rangeMask = (alpha >= np.full_like(alpha, lower)) & (alpha <= np.full_like(alpha, upper))
    alpha, y, yErr = alpha[rangeMask], y[rangeMask], yErr[rangeMask]

    # fit data
    param, paramErr = chisq_fit(double_gauss_fn, alpha, y, yErr, p0=(5, 40, 0.85, 0.05, 50, 0.97, 0.05))
    params[i] = param
    paramsErr[i] = paramErr

    fig, ax = plt.subplots()

    plot_data_fit(fig, ax, alpha, y, yErr, param)
    ax.legend()
    ax.minorticks_on()
    ax.grid(which='both')

    ax.set_title('Linien bei $\\omega_G=%.1f°$' % omegaG[i])
    ax.set_xlabel('Position $\\alpha$/°')
    ax.set_ylabel('Intensität $I$/%')
    fig.savefig(outFilenames[i])

print(params)

# paramsParamsErr = np.zeros((params.shape[0], 1+params.shape[1]+paramsErr.shape[1]))
# paramsParamsErr[:, 0] = I
# paramsParamsErr[:, 1::2] = params
# paramsParamsErr[:, 2::2] = paramsErr
# csvFilename = 'p402/data/line_params.csv'
# # np.savetxt(csvFilename, paramsParamsErr, delimiter=',')
# pd.DataFrame(paramsParamsErr)
# # paramsParamsErr.tofile('p401/data/zeeman_params.csv', sep=',', format='%.3f')
# # paramsParamsErr.tofile('p401/data/zeeman_params.csv', sep=',')

# paramsHeader = (r'$I/$A', r'$\mu_1/$°', r'$\Delta\mu_1/$°', r'$\mu_2/$°', r'$\Delta\mu_2/$°', r'$\mu_3/$°', r'$\Delta\mu_3/$°')
# paramsFrame = pd.DataFrame({
#     paramsHeader[i]: paramsParamsErr[:, i] for i in range(len(paramsHeader))
# })
# paramsFrame.to_csv(csvFilename, index=False, float_format='%.5f')

