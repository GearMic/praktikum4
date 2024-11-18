import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helpers import *


def calc_lattice_constant(lbda, alpha, beta, alphaErr, betaErr, k=1):
    gamma = np.sin(beta)+np.sin(alpha) # helper variable
    g = k*lbda / gamma
    gErr = k*lbda * np.sqrt((np.cos(beta)*betaErr/gamma**2)**2 + (np.cos(alpha)*alphaErr/gamma**2)**2)
    return g, gErr

def fit_lattice_constant_plot(filenamePlot, lbda, alpha, beta, alphaErr, betaErr, k=1):
    lbda*=k # TODO: handle this better
    fig, ax = plt.subplots()
    x = (np.sin(beta)+np.sin(alpha))
    # xErr = x**2 * np.sqrt((np.cos(beta)*betaErr)**2 + (np.cos(alpha)*alphaErr)**2)
    xErr = np.sqrt((np.sin(beta)*betaErr)**2 + (np.sin(alpha)*alphaErr)**2)
    params, paramsErr = chisq_fit(linear_fn, x, lbda, absolute_sigma=False)
    print('linear fit:', params*1e9, paramsErr*1e9)

    xFit = array_range(x)
    yFit = linear_fn(xFit, *params)

    ax.plot(xFit, yFit*1e9, label='Ausgleichsgerade', color='xkcd:red')
    ax.errorbar(x, lbda*1e9, 0, xErr, fmt='x', label='Messdaten', color='xkcd:blue')
    ax.set_xlabel(r'$\sin(\alpha)+\sin(\beta)$')
    ax.set_ylabel(r'$\lambda/\mathrm{nm}$')
    ax.grid()
    ax.legend()
    fig.savefig(filenamePlot)

    return params[1], paramsErr[1]

def get_lattice_constant(filenameIn, filenameOut, filenamePlot):
    data = pd.read_csv(filenameIn, sep=r',\s+', engine='python')
    omegaG = np.array(data[r'$\omega_G/°$'])
    omegaGerr = 0.6
    omegaB = 140
    lbda = np.array(data[r'$\lambda/\si{nm}$'])*1e-9

    alpha = np.deg2rad(omegaG)
    alphaErr = np.deg2rad(omegaGerr)
    beta = np.deg2rad(180+omegaG-omegaB)
    betaErr = alphaErr

    g, gErr = calc_lattice_constant(lbda, alpha, beta, alphaErr, betaErr)

    dataAdd = pd.DataFrame({
        r'$\alpha/°$': np.rad2deg(alpha), r'$\beta/°$': np.rad2deg(beta),
        r'$g/\si{\nm}$': g*1e9, r'$\Delta g/\si{\nm}$': gErr*1e9
    })
    data = pd.concat((data, dataAdd), axis=1)
    data.to_csv(filenameOut, index=False)

    # g, gErr = g[slice], gErr[slice]
    # g, gErr = g[:-3], gErr[:-3]
    # gMean = np.mean(g)
    # gMeanErr = np.sqrt(np.sum(gErr**2))/len(g)
    # return gMean, gMeanErr
    slicer = slice(-3)
    lbda, alpha, beta, = lbda[slicer], alpha[slicer], beta[slicer]

    gResult, gResultErr = fit_lattice_constant_plot(filenamePlot, lbda, alpha, beta, alphaErr, betaErr)
    return gResult, gResultErr



# print(get_lattice_constant('p402/data/balmer_Hg.csv'))