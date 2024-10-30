import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as optimize


def load_data(filename, errorPortion, errorMin):
    data = pd.read_csv(filename, sep='\t')
    I = np.array(data['I_A1 / A'])
    B = np.array(data['B_B1 / mT'])
    Ierr = np.maximum(errorPortion*I, errorMin[0])
    Berr = np.maximum(errorPortion*B, errorMin[1])

    return I, B, Ierr, Berr

def fit_data(I, B, Ierr, Berr):
    def fit_fn(I, a, b, c):
        return a + b*I + c*I**2

    popt, pcov = optimize.curve_fit(fit_fn, I, B, p0=(0,1,-1), sigma=Berr)
    params, paramsErr = popt, np.sqrt(np.diag(pcov))

    return params, paramsErr

def plot_data_fit(fig, ax, I, B, Ierr, Berr):
    ax.errorbar(I, B, Berr, Ierr, label='Kalibrierungsdaten Anfang')


I, B, Ierr, Berr = load_data('data/Magnetkalibrierung.txt', .02, (0.002, 0.2))
params, paramsErr = fit_data(I, B, Ierr, Berr)
print(params, paramsErr)

fig, ax = plt.subplots()
plot_data_fit(fig, ax, I, B, Ierr, Berr)
fig.savefig('plot/magnetkalib.pdf')