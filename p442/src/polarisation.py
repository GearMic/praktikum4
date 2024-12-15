from helpers import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_data(filename):
    rawData = read_data_pd(filename)
    alpha = np.deg2rad(rawData['phi/deg'])
    U = rawData['U/mV']

    return alpha, U

alphaErr = np.deg2rad(1)
Uerr = 0.8
alpha, U = load_data('p442/data/5.4polarisation.csv')

def malus_generalized(B, alpha):
    alpha0, Umin, Umax = B
    return Umin + (Umax-Umin) * np.cos(alpha-alpha0)**2


params, paramsErr = odr_fit(malus_generalized, alpha, U, 3, alphaErr, Uerr, p0=(2, 0, 10))

xFit = array_range(alpha)
yFit = malus_generalized(params, xFit)

plt.errorbar(np.rad2deg(alpha), U, Uerr, np.rad2deg(alphaErr), fmt='x', label='Messdaten der Photodiode')
plt.plot(np.rad2deg(xFit), yFit, label='Anpassung nach Malus')
plt.grid()
plt.xlim(-10, 350)
plt.ylim(-1, 12)
plt.xlabel(r'$\alpha/\degree$')
plt.ylabel(r'$U/\mathrm{mV}$')
plt.legend(loc='upper center')
plt.savefig('p442/plot/5.4polarisation.pdf')

print('U', params[1:], paramsErr[1:])
print('U corrected', params[1:]-params[1])
print('alpha0', np.rad2deg(params[0]), np.rad2deg(paramsErr[0]))

alpha0, Umin, Umax = params
alpha0Err, UminErr, UmaxErr = paramsErr

pg = (Umax-Umin)/(Umax+Umin)
pgErr = 2/(Umax+Umin)**2 * np.sqrt( (Umin*UmaxErr)**2 + (Umax*UminErr)**2 )
print('pg', pg, '+-', pgErr)