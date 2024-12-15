from helpers import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_data(filename):
    rawData = read_data_pd(filename)
    zmz0 = np.array(rawData['(z-z_0)/cm'])
    d = np.array(rawData['d/mm'])

    return zmz0, d

def preprocess_data():
    z = zmz0 + z0
    zErr = np.sqrt(zmz0Err**2 + z0Err**2)

    dataFrame = pd.DataFrame({
        r'$(z-z_0)/\si{cm}$': zmz0,
        r'$z/\si{\cm}$': z,
        r'$d/\si{\mm}$': d,
    })

    return z, zErr, dataFrame

def gauss_beam_width(B, z):
    gamma = B[0]
    w0 = np.sqrt( lbda*1e-9/np.pi * np.sqrt(L*(R-L))*1e-2 )
    zR = np.pi * w0**2 / (lbda*1e-9) * 1e2
    return gamma * w0*1e3 * np.sqrt(1 + (z/zR)**2)

def beam_fit_plot(filename, measureLabel='gemessene Strahlbreiten', fitLabel='Anpassung', fitLineLowerBound=None):
    params, paramsErr = odr_fit(gauss_beam_width, z, d, 1, zErr, dErr, p0=(1.5,))
    gamma, gammaErr = params[0], paramsErr[0]
    print('gamma', gamma, gammaErr)

    plt.errorbar(z, d, dErr, zErr, ',', label=measureLabel, zorder=5)
    if not (fitLineLowerBound is None):
        z[0] = fitLineLowerBound
    xFit = array_range(z)
    yFit = gauss_beam_width(params, xFit)
    plt.plot(xFit, yFit, label=fitLabel)

    plt.xlabel(r'$z/\mathrm{cm}$')
    plt.ylabel(r'$W/\mathrm{mm}$')
    plt.xlim(left=0, right=60)
    plt.grid()
    plt.legend(loc='upper left')

    

## short resonator
L = 48.6
Lerr = 0.1
R = 100
lbda = 632

z0 = 44.1
z0Err = 0.2
zmz0Err = 0.5
dErr = 0.1
zmz0, d = load_data('p442/data/5.5strahlA.csv')

z, zErr, dataFrame = preprocess_data()
dataFrame.to_latex('p442/data/5.5beamA.csv', index=False)
print('zmz0, z0, z, d Err:', zmz0Err, z0Err, zErr, dErr)

beam_fit_plot('p442/plot/5.5beamA.pdf', 'Messung kurzer Resonator', 'Anpassung kurzer Resonator')


## long resonator
L = 61.5
Lerr = 0.1
R = 100
lbda = 632

z0 = 44.1
z0Err = 0.2
zmz0Err = 0.5
dErr = 0.1
zmz0, d = load_data('p442/data/5.5strahlB.csv')

z, zErr, dataFrame = preprocess_data()
dataFrame.to_latex('p442/data/5.5beamB.csv', index=False)

filename = 'p442/plot/5.5beamB.pdf'
beam_fit_plot(filename, 'Messung langer Resonator', 'Anpassung langer Resonator', 2)
plt.grid()
plt.savefig(filename)