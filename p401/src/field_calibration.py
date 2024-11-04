import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as optimize

def array_range(array, overhang=0.05, nPoints=100):
    min = np.min(array)
    max = np.max(array)
    span = max-min
    min -= overhang*span
    max += overhang*span
    return np.linspace(min, max, nPoints)

def load_data(filename, errorPortion, errorMin):
    data = pd.read_csv(filename, sep='\t')
    I = np.array(data['I_A1 / A'])
    B = np.array(data['B_B1 / mT'])
    Ierr = np.maximum(errorPortion*I, errorMin[0])
    Berr = np.maximum(errorPortion*B, errorMin[1])

    return I, B, Ierr, Berr

def field_fit_fn(I, a, b, c, d):
    return a + b*I + c*I**2 + d*I**3

def fit_field_data(I, B, Ierr, Berr):

    # popt, pcov = optimize.curve_fit(field_fit_fn, I, B, p0=(0,1,-1, 0), sigma=Berr)
    popt, pcov = optimize.curve_fit(field_fit_fn, I, B, sigma=Berr)
    params, paramsErr = popt, np.sqrt(np.diag(pcov))

    return params, paramsErr

def plot_data_fit(fig, ax, I, B, Ierr, Berr, dataLabel, fitLabel):
    ax.errorbar(I, B, Berr, Ierr, label=dataLabel)

    params, paramsErr = fit_field_data(I, B, Ierr, Berr)
    xFit = array_range(I, overhang=0)
    yFit = field_fit_fn(xFit, *params)
    ax.plot(xFit, yFit, label=fitLabel)



fig, ax = plt.subplots()
I, B, Ierr, Berr = load_data('p401/data/Magnetkalibrierung.txt', .02, (0.002, 0.2))
plot_data_fit(fig, ax, I, B, Ierr, Berr, 'Daten Anfang', 'Kalibrierungskurve Anfang')
ax.legend()
ax.set_xlabel(r'Strom $I/\mathrm{A}$')
ax.set_ylabel(r'Magnetfeld $B/\mathrm{mT}$')
fig.savefig('p401/plot/magnetkalib.pdf')

fig.clf()
I, B, Ierr, Berr = load_data('p401/data/Magnetkalibrierung2.txt', .02, (0.002, 0.2))
plot_data_fit(fig, ax, I, B, Ierr, Berr, 'Daten Ende', 'Kalibrierungskurve Ende')
ax.legend()
ax.set_xlabel(r'Strom $I/\mathrm{A}$')
ax.set_ylabel(r'Magnetfeld $B/\mathrm{mT}$')
fig.savefig('p401/plot/magnetkalib2.pdf')
