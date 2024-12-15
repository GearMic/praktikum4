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

    betaErr = 1/d/(1*(x/d)**2)
    cosBeta = np.cos(beta)

    return cosBeta, np.abs(np.sin(beta)*betaErr)

n, x, xErr = load_data('p442/data/5.3lambda.csv')
nAbs = np.abs(n)
cosx, cosxErr = pos_to_cos(n, x, xErr)
params, paramsErr = odr_fit(linear_fn, cosx, nAbs, 2, xErr=cosxErr, p0=(1, -1))

xFit = array_range(cosx)
yFit = linear_fn_odr(params, xFit)

plt.errorbar(cosx, nAbs, 0, cosxErr)
plt.plot(xFit, yFit)
plt.xlabel('cos(beta)')
plt.ylabel('|n|')
plt.savefig('p442/plot/5.3lambda.pdf')


print(params, paramsErr)