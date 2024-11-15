import numpy as np
import pandas as pd


def calc_lattice_constant(lbda, alpha, beta, alphaErr, betaErr, k=1):
    gamma = np.sin(beta)+np.sin(alpha) # helper variable
    g = k*lbda / gamma
    gErr = k*lbda * np.sqrt((np.cos(beta)*betaErr/gamma**2)**2 + (np.cos(alpha)*alphaErr/gamma**2)**2)
    return g, gErr

# def get_lattice_constant(filename, slice=slice()):
def get_lattice_constant(filename):
    data = pd.read_csv(filename, sep=r',\s+', engine='python')
    omegaG = np.array(data[r'\omega_G/°'])
    omegaGerr = 0.6
    omegaB = 140
    lbda = np.array(data[r'\lambda/\si{nm}'])*1e-9

    alpha = np.deg2rad(omegaG)
    alphaErr = np.deg2rad(omegaGerr)
    beta = np.deg2rad(180+omegaG-omegaB)
    betaErr = alphaErr

    g, gErr = calc_lattice_constant(lbda, alpha, beta, alphaErr, betaErr)
    # g, gErr = g[slice], gErr[slice]
    g, gErr = g[:-3], gErr[:-3]
    gMean = np.mean(g)
    gMeanErr = np.sqrt(np.sum(gErr**2))/len(g)
    # TODO: better way to get one value from multiple? (the error calculation is wrong here)

    return gMean, gMeanErr


# print(get_lattice_constant('p402/data/balmer_Hg.csv'))