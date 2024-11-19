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

def load_data(filename, yErr):
    data = pd.read_csv(filename, sep='\t', encoding_errors='replace')
    alpha = np.array(data.iloc[:, 0])
    y = np.array(data.iloc[:, 1])
    yErr = np.array(np.full_like(y, yErr))
    # yErr = np.maximum(y*errorPortion, errorMin)

    return alpha, y, yErr

def gauss_fn(x, a, b, c):
    return a * np.exp(-(x-b)**2/ (2*c**2))

def triple_gauss_fn(alpha, a1, b1, c1, a2, b2, c2, a3, b3, c3, B):
    return B + gauss_fn(alpha, a1, b1, c1) + gauss_fn(alpha, a2, b2, c2) + gauss_fn(alpha, a3, b3, c3)

def fit_ccd_data(alpha, y, yErr, p0=None):
    popt, pcov = optimize.curve_fit(triple_gauss_fn, alpha, y, sigma=yErr, p0=p0, maxfev=10000)
    # popt, pcov = optimize.curve_fit(gauss_fn, alpha, y, sigma=yErr, maxfev=10000)
    params, paramsErr = popt, np.sqrt(np.diag(pcov))

    return params, paramsErr

def linear_fn(x, a, b):
    return a + b*x

def fit_energy_split(B, deltaE, deltaEerr, p0=None):
    popt, pcov = optimize.curve_fit(linear_fn, B, deltaE, sigma=deltaEerr, p0=p0, maxfev=10000)
    params, paramsErr = popt, np.sqrt(np.diag(pcov))

    return params, paramsErr

def plot_data_fit(fig, ax, alpha, y, yErr, params=None):
    ax.errorbar(alpha, y, yErr, fmt='x', label='Daten')
    # ax.plot(alpha, y, '-', lw=1, label='Daten')

    # paramsPrint = np.array((params, paramsErr)).T
    # paramsPrint = (r' \pm '.join(tuple(np.array(param, dtype=str))) for param in paramsPrint)
    # paramsPrint = r',\ '.join(paramsPrint)
    # print(paramsPrint)

    if not (params is None):
        xFit = array_range(alpha, overhang=0)
        yFit = triple_gauss_fn(xFit, *params)
        ax.plot(xFit, yFit, label='Kalibrierungskurve')

"""
def alpha_to_lambda(alpha, alphaErr):
    alpha *= 2*np.pi/360 # convert to radians
    d = 0.004
    k = 2
    n = 1.457
    lbda = 2*d/k * np.sqrt(n**2-(np.sin(alpha)**2))
    lbdaErr = d/k * (np.sin(2*alpha)**2) / np.sqrt(n**2 - np.sin(alpha)**2) * alphaErr
    return lbda, lbdaErr

def calc_delta_E(lbda1, lbda1, lbda1Err, lbda2Err):
    lbda1, lbda2, lbda3 = lbda
    lbda1Err, lbda2Err, lbda3Err = lbdaErr
    # lbdaDiff1 = np.abs(lbda2 - lbda1)
    # lbdaDiff1Err = np.sqrt(lbda2Err**2 + lbda1Err**2)
    # lbdaDiff2 = np.abs(lbda3 - lbda2)
    # lbdaDiff2Err = np.sqrt(lbda3Err**2 + lbda2Err**2)

    # deltaLbda = (lbdaDiff1 + lbdaDiff2)/2
    # deltaLbdaErr = np.sqrt(lbdaDiff1Err**2 + lbdaDiff2Err**2)/2
"""

def calc_delta_E(alphaPi, alphaSigma, alphaPiErr, alphaSigmaErr):
    n = 1.457
    h = 6.626e-34
    c = 3e8
    lbda0 = 643.8e-9
    alphaPi *= 2*np.pi/360
    alphaSigma *= 2*np.pi/360
    alphaPiErr *= 2*np.pi/360
    alphaSigmaErr *= 2*np.pi/360

    gamma = -c/lbda0 * h
    numerator = np.sqrt(n**2 - np.sin(alphaPi)**2)
    denominator = np.sqrt(n**2 - np.sin(alphaSigma)**2)
    deltaE = gamma * (1 - numerator/denominator)
    # deltaEerr = np.sqrt(alphaPiErr**2 + alphaSigmaErr**2)/7e19
    deltaEerr = gamma * np.sqrt((
        alphaPiErr * np.sin(alphaPi)*np.cos(alphaPi)/numerator/denominator)**2 +
        (alphaSigmaErr * np.sin(alphaSigma)*np.cos(alphaSigma)*numerator/denominator**3)**2)
    # deltaEerr *= 3

    return deltaE, deltaEerr

def mean_mean_err(value1, value2, value1Err, value2Err):
    mean = (value1 + value2) / 2
    meanErr = np.sqrt(value1Err**2 + value2Err**2) / 2
    return mean, meanErr

# alpha, y, yErr, load_data('p401/data/interference_9.1A.txt', 0.02, 0.5)

alpha, y, yErr = load_data('p401/data/interference_9.1A.txt', 1)
fig, ax = plt.subplots()
plot_data_fit(fig, ax, alpha, y, yErr)
ax.minorticks_on()
ax.grid(which='both')
fig.savefig('p401/plot/gauss_1.pdf')

I = np.array((2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.5, 8.0, 8.5, 9.1))
alphaRange = np.array(((0.77, 1.16), (0.77, 1.16), (0.77, 1.16), (0.77, 1.16), (0.77, 1.16), (0.77, 1.16), (0.77, 1.16), (0.77, 1.16), (0.77, 1.16), (0.77, 1.16)))
inFilenames = tuple('p401/data/interference_%.1fA.txt' % Ival for Ival in I)
outFilenames = tuple('p401/plot/gauss_%.1fA.pdf' % Ival for Ival in I)

params, paramsErr = np.zeros((len(I), 10)), np.zeros((len(I), 10))
for i in range(len(inFilenames)):
# for i in range(9, 10):
    lower, upper = alphaRange[i, 0], alphaRange[i, 1]

    alpha, y, yErr = load_data(inFilenames[i], 1)
    rangeMask = (alpha >= np.full_like(alpha, lower)) & (alpha <= np.full_like(alpha, upper))
    alpha, y, yErr = alpha[rangeMask], y[rangeMask], yErr[rangeMask]

    # fit data
    param, paramErr = fit_ccd_data(alpha, y, yErr, p0=(40, 0.85, 0.05, 50, 0.97, 0.05, 40, 1.09, 0.05, 1))
    params[i] = param
    paramsErr[i] = paramErr

    fig, ax = plt.subplots()

    plot_data_fit(fig, ax, alpha, y, yErr, param)
    ax.legend()
    ax.minorticks_on()
    ax.grid(which='both')

    ax.set_title('Intensitätsmaxima für $I=%.1f\\mathrm{A}$' % I[i])
    ax.set_xlabel('Position $\\alpha$/°')
    ax.set_ylabel('Intensität $I$/%')
    fig.savefig(outFilenames[i])

def field_fit_fn(I, a, b, c, d):
    return a + b*I + c*I**2 + d*I**3

a, b, c, d = -0.326, 56.82, 3.960, -0.5288
aErr, bErr, cErr, dErr = 0.019, 0.28, 0.099, 0.0082

print('params', params[-1])
print('paramsErr', paramsErr[-1])

paramsParamsErr = np.zeros((params.shape[0], 1+params.shape[1]+paramsErr.shape[1]))
paramsParamsErr[:, 0] = I
paramsParamsErr[:, 1::2] = params
paramsParamsErr[:, 2::2] = paramsErr
csvFilename = 'p401/data/zeeman_params.csv'
# np.savetxt(csvFilename, paramsParamsErr, delimiter=',')
pd.DataFrame(paramsParamsErr)
# paramsParamsErr.tofile('p401/data/zeeman_params.csv', sep=',', format='%.3f')
# paramsParamsErr.tofile('p401/data/zeeman_params.csv', sep=',')

paramsHeader = (r'$I/$A', r'$\mu_1/$°', r'$\Delta\mu_1/$°', r'$\mu_2/$°', r'$\Delta\mu_2/$°', r'$\mu_3/$°', r'$\Delta\mu_3/$°')
paramsFrame = pd.DataFrame({
    paramsHeader[i]: paramsParamsErr[:, i] for i in range(len(paramsHeader))
})
paramsFrame.to_csv(csvFilename, index=False, float_format='%.5f')

alpha_sigma_minus = params[:, 1]
alpha_pi = params[:, 4]
alpha_sigma_plus = params[:, 7]
alpha_sigma_minus_err = paramsErr[:, 1]
alpha_pi_err = paramsErr[:, 4]
alpha_sigma_plus_err = paramsErr[:, 7]
deltaEMinus, deltaEMinusErr = calc_delta_E(alpha_pi, alpha_sigma_minus, alpha_pi_err, alpha_sigma_minus_err)
deltaEPlus, deltaEPlusErr = calc_delta_E(alpha_pi, alpha_sigma_plus, alpha_pi_err, alpha_sigma_plus_err)
deltaE, deltaEErr = mean_mean_err(np.abs(deltaEMinus), np.abs(deltaEPlus), deltaEMinusErr, deltaEPlusErr)

B = field_fit_fn(I, a, b, c, d)
Berr = np.sqrt(field_fit_fn(I, a+aErr, b+bErr, c+cErr, d+dErr)**2 + field_fit_fn(I, a-aErr, b-bErr, c-cErr, d-dErr)**2)/9/I
Ierr = 0.2
Berr = np.sqrt(aErr**2 +bErr**2 *I**2 + cErr**2*I**4 + dErr**2*I**6 + Ierr**2*(b**2 + 4*c**2*I**2 + 9*d**2*I**4))
e=1.602e-19

energyData = pd.DataFrame({
    "B": B, "Berr": Berr,
    # "dEm": deltaEMinus/e, 'dEm_err': deltaEMinusErr/e,
    # "dEp": deltaEPlus/e, 'dEp_err': deltaEPlusErr/e,
    "dE": deltaE/e, 'dE_err': deltaEErr/e
})
energyData.to_csv('p401/data/energy_data.csv', index=False)

sliceFront = 3
B = B[sliceFront:]/1000
Berr = Berr[sliceFront:]/1000
deltaE = deltaE[sliceFront:]
deltaEErr = deltaEErr[sliceFront:]

fig, ax = plt.subplots()
ax.errorbar(B, deltaE, deltaEErr, Berr, fmt=',', label='Messdaten')
ax.set_xlabel("B/T")
ax.set_ylabel(r"$\Delta$E/J")


params, paramsErr = fit_energy_split(B, deltaE, deltaEErr)
xFit = array_range(B)
yFit = linear_fn(xFit, *params)
ax.plot(xFit, yFit, label='Ausgleichsgerade')
ax.grid()
ax.legend()
fig.savefig('p401/plot/energy_plot.pdf')
print(params,paramsErr)
print(params,paramsErr*2.3)
