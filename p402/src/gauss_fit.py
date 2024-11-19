import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as optimize
from helpers import *

def load_data(filename, errorPortion, errorMin):
    data = pd.read_csv(filename, sep='\t', encoding_errors='replace', decimal=',')
    # alpha = np.array(data.iloc[:, 0], dtype=float)
    # y = np.array(data.iloc[:, 1], dtype=float)
    # yErr = np.array(np.full_like(y, yErr), dtype=float)
    alpha = np.array(data.iloc[:, 0])
    y = np.array(data.iloc[:, 1])
    # yErr = np.array(np.full_like(y, yErr))
    yErr = np.maximum(y*errorPortion, errorMin)
    # print(alpha.dtype, y.dtype, yErr.dtype)

    return alpha, y, yErr

def gauss_fn(x, a, b, c):
    return a * np.exp(-(x-b)**2/ (2*c**2))

def double_gauss_fn(alpha, a1, mu1, sigma1, a2, mu2, sigma2, B):
    return B + gauss_fn(alpha, a1, mu1, sigma1) + gauss_fn(alpha, a2, mu2, sigma2)

def plot_data_fit(fig, ax, alpha, y, yErr, params=None):
    ax.errorbar(alpha, y, yErr, fmt='-', label='Daten', lw=0.5)
    # ax.plot(alpha, y, '-', lw=1, label='Daten')

    # paramsPrint = np.array((params, paramsErr)).T
    # paramsPrint = (r' \pm '.join(tuple(np.array(param, dtype=str))) for param in paramsPrint)
    # paramsPrint = r',\ '.join(paramsPrint)
    # print(paramsPrint)

    if not (params is None):
        xFit = array_range(alpha, overhang=0)
        yFit = double_gauss_fn(xFit, *params)
        ax.plot(xFit, yFit, label='Kalibrierungskurve')

def full_gauss_fit_for_lines():
    omegaG = np.array((13.8, 18.2, 37.0))
    alphaRange = np.array(((-0.5, 0.9), (-0.06, 0.04), (-0.2, 0.1)))
    p0 = ((20, -0.02, 0.05, 20, 0, 0.05, 40), (10, -0.03, 0.01, 20, 0, 0.02, 5), (20, -0.1, 0.02, 80, 0, 0.02, 5))
    doFit = (False, True, True)
    inFilenames = tuple('p402/data/ccd/line%.1f.txt' % omega for omega in omegaG)
    outFilenames = tuple('p402/plot/line%.1f.pdf' % omega for omega in omegaG)
    lowerBounds = np.array((0, -np.inf, 0, 0, -np.inf, 0, 0))
    upperBounds = np.array((np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf))

    params, paramsErr = np.zeros((len(omegaG), 7)), np.zeros((len(omegaG), 7))
    for i in range(len(inFilenames)):
    # for i in range(9, 10):
        lower, upper = alphaRange[i, 0], alphaRange[i, 1]

        alpha, y, yErr = load_data(inFilenames[i], 0.0001, 0.2)
        rangeMask = (alpha >= np.full_like(alpha, lower)) & (alpha <= np.full_like(alpha, upper))
        alpha, y, yErr = alpha[rangeMask], y[rangeMask], yErr[rangeMask]

        # fit data
        param, paramErr = None, None
        if doFit[i]:
            param, paramErr = chisq_fit(
                double_gauss_fn, alpha, y, yErr, p0=p0[i],
                bounds = (lowerBounds, upperBounds))

            params[i] = param
            paramsErr[i] = paramErr
        

        fig, ax = plt.subplots()

        plot_data_fit(fig, ax, alpha, y, yErr, param)
        ax.legend()
        ax.minorticks_on()
        ax.grid(which='both')

        ax.set_title('Linien bei $\\omega_G=%.1f°$' % omegaG[i])
        ax.set_xlabel(r'Position $\gamma$/°')
        ax.set_ylabel(r'Intensität $I$/\%')
        fig.savefig(outFilenames[i])

    mu1, mu1Err = params[:, 1], paramsErr[:, 1]
    mu2, mu2Err = params[:, 4], paramsErr[:, 4]
    deltaBeta = np.abs(mu2 - mu1)
    deltaBetaErr = np.sqrt(mu1Err**2 + mu2Err**2)

    omegaB = 140
    beta = (180+omegaG-omegaB)
    betaErr = 0.6
    alpha = omegaG

    paramsFrame = pd.DataFrame({
        r'$\alpha/°$': alpha, r'$\beta/°$': beta,
        r'$\mu_1/°$': mu1, r'$\Delta\mu_1/°$': mu1Err, r'$\mu_2/°$': mu2, r'$\Delta\mu_2/°$': mu2Err,
        r'$\delta\beta/°$': deltaBeta, r'$\Delta\delta\beta/°$': deltaBetaErr, 
    })
    paramsFrame.to_csv('p402/data/balmer_gauss_fit.csv', index=False)

    return alpha, beta, deltaBeta, betaErr, deltaBetaErr, paramsFrame