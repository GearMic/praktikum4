import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.constants import h, c, e
from lattice_constant import get_lattice_constant
from gauss_fit import full_gauss_fit_for_lines
from helpers import *

def calc_lbda(g, alpha, beta, gErr, alphaErr, betaErr, k=1):
    gamma = (np.sin(alpha)+np.sin(beta))
    lbda = g/k * gamma
    lbdaErr = 1/k * np.sqrt( (gErr*gamma)**2 + g**2 * ((np.cos(beta)*betaErr)**2 + (np.cos(alpha)*alphaErr)**2) )

    return lbda, lbdaErr

def calc_E(lbda, lbdaErr):
    E = h*c/lbda
    Eerr = h*c*lbdaErr/lbda**2
    return E, Eerr

def calc_lbdarec(lbda, lbdaErr):
    lbdarec = 1/lbda
    lbdarecErr = lbdaErr/lbda**2
    return lbdarec, lbdarecErr

def calc_delta_lbda(beta, betaErr, deltaBeta):
    # calculate Aufspaltung
    deltaLambda = g * np.cos(beta) * deltaBeta
    deltaLambdaErr = deltaBeta * np.sqrt((gErr*np.cos(beta))**2 + (g*np.sin(beta)*betaErr)**2)
    return deltaLambda, deltaLambdaErr

def get_isotropy_data(beta, betaErr, deltaBeta, m, lbda, lbdaErr):
    deltaLbda, deltaLbdaErr = calc_delta_lbda(beta, betaErr, deltaBeta)

    isotropyData = pd.DataFrame({
        r'$m$': m, r'$\lambda/\si{\nm}$': lbda*1e9, r'$\Delta\lambda/\si{\nm}$': lbdaErr*1e9,
        r'$\delta\lambda/\si{\nm}$': deltaLbda*1e9, r'$\Delta\delta\lambda/\si{\nm}$': deltaLbdaErr*1e9,
    })
    return isotropyData


g, gErr = get_lattice_constant('p402/data/balmer_Hg.csv')

data = pd.read_csv('p402/data/balmer_H.csv', sep=r',\s+', engine='python')
omegaG = np.array(data[r'\omega_G/°'])
omegaGerr = 0.6
d = (np.array(data[r'd/\si{\mm}'])-5) / 1e3
dErr = 0.1 / 1e3
omegaB = 140
m = np.array(data['m'])

f = 0.3
alpha = np.deg2rad(omegaG)
alphaErr = np.deg2rad(omegaGerr)
beta = np.deg2rad(180+omegaG-omegaB)
betaErr = alphaErr

# reshape data
deltaBeta = (d[1::2] - d[::2])/f
slicer = slice(None, None, 2)
alpha, beta, m = alpha[slicer], beta[slicer], m[slicer]
lbda, lbdaErr = calc_lbda(g, alpha, beta, gErr, alphaErr, betaErr)

# get rydberg constant
y, yErr = calc_lbdarec(lbda, lbdaErr)
x = 1/m**2

# slicer = slice(2, None, 2)
# y, yErr, x = y[slicer], yErr[slicer], x[slicer]
mask = m!=0
y, yErr, x = y[mask], yErr[mask], x[mask]

params, paramsErr = chisq_fit(linear_fn, x, y, yErr)

fig, ax = plt.subplots()
ax.errorbar(x, y, yErr, fmt='x', color='xkcd:blue', label='Messdaten')
xFit = array_range(x)
yFit = linear_fn(xFit, *params)
ax.plot(xFit, yFit, color='xkcd:red', label='Ausgleichsgerade')

ax.set_xlabel(r'$1/m^2$')
ax.set_ylabel(r'$\frac{1}{\lambda}\,/\,\frac{1}{\mathrm{m}}$')
fig.savefig('p402/plot/rydberg_fit.pdf')


# get wavelengths and splitting for ocular data
isotropyDataOcular = get_isotropy_data(beta, betaErr, deltaBeta, m, lbda, lbdaErr)
isotropyDataOcular.to_csv('p402/data/isotropy_ocular.csv', index=False)

alpha, beta, deltaBeta, betaErr, deltaBetaErr, gaussFrame = full_gauss_fit_for_lines()
alphaErr = betaErr
lbda, lbdaErr = calc_lbda(g, alpha, beta, gErr, alphaErr, betaErr)
isotropyDataCCD = get_isotropy_data(beta, betaErr, deltaBeta, m[mask], lbda, lbdaErr)
isotropyDataCCD.to_csv('p402/data/isotropy_CCD.csv', index=False)

