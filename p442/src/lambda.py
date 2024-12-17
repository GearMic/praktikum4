from helpers import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import scipy.optimize as optimize

def load_data(filename):
    rawData = read_data_pd(filename)
    n = rawData['m']
    x = rawData['x_m/cm']
    xErr = rawData['x_mErr/cm']

    return n, x, xErr

g = 1e-9/600
d = 28.25
dErr = 0.5
alpha = 0

def pos_to_cos(n, x, xErr):
    x = np.maximum(x, 0.0001)
    beta = np.arctan(x/d)

    betaErr = np.minimum(1/d/(1*(x/d)**2), 0.1)
    cosBeta = np.cos(beta)

    return cosBeta, np.abs(np.sin(beta)*betaErr)

#fit 1
n, x, xErr = load_data('p442/data/5.3lambda.csv')
nAbs = np.abs(n)
cosx, cosxErr = pos_to_cos(n, x, xErr)
params, paramsErr = odr_fit(linear_fn_odr, cosx, nAbs, 2, xErr=cosxErr, p0=(1, -1))

xFit = array_range(cosx)
yFit = linear_fn_odr(params, xFit)

plt.errorbar(cosx, nAbs, 0, cosxErr)
plt.plot(xFit, yFit)
plt.xlabel('cos(beta)')
plt.ylabel('|n|')
plt.savefig('p442/plot/5.3lambda.pdf')

print(params, paramsErr)

# fit 2
params, paramsErr = odr_fit(linear_fn_odr, nAbs, cosx, 2, yErr=cosxErr, p0=(1, -1))

xFit = array_range(nAbs)
yFit = linear_fn_odr(params, xFit)

plt.clf()
plt.errorbar(nAbs, cosx, cosxErr)
plt.plot(xFit, yFit)
plt.xlabel('|n|')
plt.ylabel('cos(beta)')
plt.savefig('p442/plot/5.3lambda2.pdf')

m = params[1]
lbda = -params[1]*g
print(params, paramsErr)
print('m, lbda', m, lbda)