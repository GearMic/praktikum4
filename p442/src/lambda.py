from helpers import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import scipy.optimize as optimize

def load_data(filename):
    rawData = read_data_pd(filename)
    n = np.array(rawData['m'])
    x = np.array(rawData['x_m/cm'])
    xErr = np.array(rawData['x_mErr/cm'])

    return n, x, xErr

g = 1e-3/600
d = 28.25
dErr = 0.5
alpha = 0

def pos_to_cos(n, x, xErr):
    # x = np.maximum(x, 0.0001)
    beta = np.arctan(x/d)

    # betaErr = np.minimum(1/d/(1*(x/d)**2), 0.1)
    # betaErr = 1/d/(1+(x/d)**2)
    betaErr = (1+(x/d)**2) * np.sqrt( (xErr/d)**2 + (dErr*x/d**2)**2 )
    sinBeta = np.sin(beta)

    return sinBeta, np.abs(np.cos(beta)*betaErr)


n, x, xErr = load_data('p442/data/5.3lambda.csv')
# nAbs = np.abs(n)
nAbs = n
cosx, cosxErr = pos_to_cos(n, x, xErr)
print('n', n)
print('sin', cosx)
print('Delta sin', cosxErr)

#fit 1
params, paramsErr = odr_fit(linear_fn_odr, cosx, nAbs, 2, xErr=cosxErr, p0=(1, -1))

xFit = array_range(cosx)
yFit = linear_fn_odr(params, xFit)

plt.errorbar(cosx, nAbs, 0, cosxErr, fmt='x')
plt.plot(xFit, yFit)
plt.xlabel('sin(beta)')
plt.ylabel('|n|')
plt.savefig('p442/plot/5.3lambda.pdf')

m, mErr = params[1], paramsErr[1]
lbda = g/m
lbdaErr = mErr * g/m**2
print('params', params, paramsErr)
print('lbda', lbda, lbdaErr)

""" # fit 2
params, paramsErr = odr_fit(linear_fn_odr, nAbs, cosx, 2, yErr=cosxErr, p0=(1, -1))

xFit = array_range(nAbs)
yFit = linear_fn_odr(params, xFit)

plt.clf()
plt.errorbar(nAbs, cosx, cosxErr, fmt='x')
plt.plot(xFit, yFit)
plt.xlabel('|n|')
plt.ylabel('sin(beta)')
plt.savefig('p442/plot/5.3lambda2.pdf')

m = params[1]
lbda = -m*g
print(params, paramsErr)
print('m, lbda', m, lbda) """