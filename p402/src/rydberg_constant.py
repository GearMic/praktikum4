import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.constants import h, c, e, epsilon_0, m_e
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

def calc_delta_lbda(beta, betaErr, deltaBeta, deltaBetaErr):
    # calculate Aufspaltung
    print(beta)
    print('cos', np.cos(beta))
    deltaLambda = g * np.cos(beta) * deltaBeta
    deltaLambdaErr = np.sqrt((deltaBetaErr*g*np.cos(beta))**2 + (deltaBeta*gErr*np.cos(beta))**2 + (deltaBeta*g*np.sin(beta)*betaErr)**2)
    return deltaLambda, deltaLambdaErr

def get_isotropy_data(beta, betaErr, deltaBeta, m, lbda, lbdaErr, deltaBetaErr):
    deltaLbda, deltaLbdaErr = calc_delta_lbda(beta, betaErr, deltaBeta, deltaBetaErr)

    isotropyData = pd.DataFrame({
        r'$m$': m, r'$\lambda/\si{\nm}$': lbda*1e9, r'$\Delta\lambda/\si{\nm}$': lbdaErr*1e9,
        r'$\delta\lambda/\si{\nm}$': deltaLbda*1e9, r'$\Delta\delta\lambda/\si{\nm}$': deltaLbdaErr*1e9,
    })
    return isotropyData

def calc_h_from_ryd(R, Rerr):
    gamma = (m_e*e**4/(8*c*epsilon_0**2))**(1/3)
    h = gamma / R**(1/3)
    hErr = h/3/R*Rerr
    return h, hErr


g, gErr = get_lattice_constant('p402/data/balmer_Hg.csv', 'p402/data/balmer_Hg_analysis.csv', 'p402/plot/fit-gitterkonstante.pdf')

data = pd.read_csv('p402/data/balmer_H.csv', sep=r',\s+', engine='python')
omegaG = np.array(data[r'$\omega_G/°$'])
omegaGerr = 0.6
d = (np.array(data[r'$d/\si{\mm}$'])-5) / 1e3
dErr = 0.1 / 1e3
omegaB = 140
m = np.array(data['$m$'])

f = 0.3
alpha = np.deg2rad(omegaG)
alphaErr = np.deg2rad(omegaGerr)
beta = np.deg2rad(180+omegaG-omegaB)
betaErr = alphaErr

# reshape data
deltaBeta = (d[1::2] - d[::2])/f
deltaBetaErr = dErr*np.sqrt(2)/f
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
rydbergMean, rydbergMeanErr = mean_mean_err(params[0]*4, np.abs(params[1]), paramsErr[0]*4, paramsErr[1])
print('rydberg fit /1e7:', params/1e7, paramsErr/1e7)
print('rydberg mean /1e7', rydbergMean/1e7, rydbergMeanErr/1e7)
print('h', calc_h_from_ryd(rydbergMean, rydbergMeanErr))

fig, ax = plt.subplots()
ax.errorbar(x, y, yErr, fmt='x', color='xkcd:blue', label='Messdaten')
xFit = array_range(x)
yFit = linear_fn(xFit, *params)
ax.plot(xFit, yFit, color='xkcd:red', label='Ausgleichsgerade')

ax.set_xlabel(r'$1/m^2$')
ax.set_ylabel(r'$\frac{1}{\lambda}\,/\,\frac{1}{\mathrm{m}}$')
ax.grid()
fig.savefig('p402/plot/rydberg_fit.pdf')


# get wavelengths and splitting for ocular data
isotropyDataOcular = get_isotropy_data(beta, betaErr, deltaBeta, m, lbda, lbdaErr, deltaBetaErr)
isotropyDataOcular.to_csv('p402/data/isotropy_ocular.csv', index=False)

alpha, beta, deltaBeta, betaErr, deltaBetaErr, gaussFrame = full_gauss_fit_for_lines()
alphaErr = betaErr
lbda, lbdaErr = calc_lbda(g, alpha, beta, gErr, alphaErr, betaErr)
isotropyDataCCD = get_isotropy_data(beta, betaErr, deltaBeta, m[mask], lbda, lbdaErr, deltaBetaErr)
isotropyDataCCD.to_csv('p402/data/isotropy_CCD.csv', index=False)

