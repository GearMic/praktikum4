from lib.helpers import *
from pathlib import Path
import matplotlib.pyplot as plt
import lmfit

pltColors = ('red', 'green', 'blue', 'yellow', 'purple')
nErr = 1
minorAlpha=0.3
majorAlpha=1.0


def Gaussian(x, a, x0, sigma, offset=0):
    """ (von Samuel)
    1-dimensional Gaussian distribution

    Parameters
    ----------
    x : np.array
        Coordinates
    a : float
        Amplitude
    x0 : float
        Center
    sigma : float
        Standard deviation
    offset : float, optional
        Absolute offset value, defaults to 0

    Returns
    -------
    np.array
    """
    gauss = a * np.exp(-0.5 * np.square((x-x0)/sigma))
    return offset + gauss

def Gaussian_linoff(x,a1,x01,sigma1,m,b):
    '''Sum of two Gaussians and an linear offset 
    '''
    return Gaussian(x,a1,x01,sigma1)+m*x+b

def Double_Gaussian_linoff(x,a1,x01,sigma1,a2,x02,sigma2,m,b):
    '''Sum of two Gaussians and an linear offset 
    '''
    return Gaussian(x,a1,x01,sigma1)+Gaussian(x,a2,x02,sigma2)+m*x+b

def load_data(file):
    data = read_data_pd(file, sep=r'\s+')
    n = np.array(data['n_1'])
    N = np.array(data['N_1'])

    return n, N

def gauss_fn(B, x):
    a, x0, sigma = B
    return a * np.exp(-0.5 * np.square((x-x0)/sigma))

def double_gauss_fn(B, x):
    return gauss_fn(B[:3], x) + gauss_fn(B[3:6], x) + B[-1]

def multi_gauss_fn(B, x):
    result = B[-1] # linear offset
    for i in range(len(B)//3):
        result += gauss_fn(B[3*i:3*(i+1)], x)
    return result

def multi_gauss_ODR_fit(x, y, nGaussians=None, xErr=None, yErr=None, p0: np.array=None):
    if nGaussians is None:
        if p0 is None:
            raise ValueError('Cannot determine amount of parameters.')
        else:
            nGaussians = nParams//3
    nParams = 3*nGaussians+1

    return odr_fit(multi_gauss_fn, x, y, nParams, xErr, yErr, p0)

def fluorescence_energy_calibration(dataFilename, plot1Filename, plot2Filename, energyValues, p0a, p0b):
    n, N = load_data(dataFilename)

    # fit gaussians to data
    nSlice, NSlice = slice_from_range(80, 160, n, N)
    paramsGauss, paramsGaussErr = multi_gauss_ODR_fit(nSlice, NSlice, 4, p0=np.concatenate((p0a[:-1], p0b)), xErr=nErr)

    # do linear fit for energy calibration
    energyValues /= 1e3
    x0 = paramsGauss[1:-1:3]
    x0Err = paramsGaussErr[1:-1:3] # TODO: is the slice done by reference?
    print('centers', x0, x0Err)
    params, paramsErr = odr_fit(linear_fn_odr, x0, energyValues, 2, xErr=x0Err)

    fig, ax = plt.subplots()
    ax.plot(n, N, label='Messdaten')
    ax.plot(*fit_curve(multi_gauss_fn, paramsGauss, nSlice, 500), zorder=4, label='Anpassung')
    ax.set_xlim(80, 160)
    ax.set_ylim(bottom=0)
    ax.minorticks_on()
    ax.grid(which='minor', visible=True, alpha=minorAlpha)
    ax.grid(which='major', visible=True, alpha=majorAlpha)
    ax.set_xlabel('$n$')
    ax.set_ylabel('$N$')
    ax.legend()
    fig.savefig(plot1Filename)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.errorbar(x0, energyValues, 0, x0Err, 'x', zorder=4, label='Maxima aus Gauß-Anpassung')
    ax.plot(*fit_curve(linear_fn_odr, params, x0), label='lineare Anpassung')
    ax.set_xlim(102, 150)
    ax.minorticks_on()
    ax.grid(which='minor', visible=True, alpha=minorAlpha)
    ax.grid(which='major', visible=True, alpha=majorAlpha)
    ax.set_xlabel('$n$')
    ax.set_ylabel(r'$E/\mathrm{keV}$')
    ax.legend()
    fig.savefig(plot2Filename)
    plt.close(fig)

    return params, paramsErr

def n_to_E(n, energyParams, energyParamsErr):
    E = linear_fn_odr(energyParams, n)
    _, b = energyParams
    aErr, bErr = energyParamsErr
    EErr = np.sqrt(aErr**2 + (bErr*n)**2 + (b*nErr)**2)
    return E, EErr

def plot_lines_directory(inDir, outDir, energyParams, linesDic={}, fitDic={}):
    """
    Plot all files from inDir and save plots in outDir.
    Use energy calibration with the parameters energyParams and
    plot vertical lines corresponding to the values in linesDic.
    """

    inDir = Path(inDir)
    sampleNames = []
    maxEnergies = [] # Photon energy corresponding to the maximum Intensity
    for file in inDir.iterdir():
        if not file.is_file():
            continue
        sampleName = file.stem
        sampleNames.append(sampleName)
        
        _, N = load_data(file)
        # E = linear_fn_odr(energyParams, n)
        maxEnergies.append(E[np.argmax(N)])

        fig, ax = plt.subplots()
        # ax.plot(E, N, label='Messdaten')

        # ax.errorbar(E, N, 0, EErr, label='Messdaten')
        ax.plot(E, N, label='Messdaten')
        if sampleName in linesDic:
            lines, lineNames = linesDic[sampleName]
            for i in range(len(lines)):
                ax.axvline(lines[i], color=pltColors[i], lw=1, label=lineNames[i])
        if sampleName in fitDic:
            params, paramsErr = multi_gauss_ODR_fit(E, N, 1, xErr=EErr, p0=fitDic[sampleName])
            ax.plot(*fit_curve(multi_gauss_fn, params, E, 500), label='Anpassung')

        ax.set_ylim(bottom=0)
        ax.set_xlabel(r'$E/\mathrm{keV}$')
        ax.set_ylabel(r'$N$')
        ax.legend()
        ax.minorticks_on()
        ax.grid(which='minor', visible=True, alpha=0.5)
        ax.grid(which='major', visible=True)
        outFilename = str(Path(outDir))+'/'+sampleName+'.pdf'
        fig.savefig(outFilename)
        plt.close(fig)

    # print('sampleNames', sampleNames)
    # print('maxEnergies', maxEnergies)
    maxEnergyDic = {sampleNames[i]: maxEnergies[i] for i in range(len(sampleNames))}
    print('maxEnergies', maxEnergyDic)

def fit_fluorescence_data(inFile, outFile, energyParams, lineData, nGaussians, **kwargs):
    """
    Fit nGaussians gaussians to data from inFile (transformed to energy using energy calibration from energyParams),
    plot the result in outFile (including vertical lines corresponding to the data from lineData) and return the gaussian parameters.
    """
    n, N = load_data(inFile)
    E = linear_fn_odr(energyParams, n)

    params, paramsErr = multi_gauss_ODR_fit(E, N, nGaussians, **kwargs)

    fig, ax = plt.subplots()

    ax.plot(E, N, label='Messdaten')
    ax.plot(*fit_curve(multi_gauss_fn, params, E, 500), label='Gauß-Anpassung')
    if not (lineData is None):
        lines, lineNames = lineData
        for i in range(len(lines)):
            ax.axvline(lines[i], color=pltColors[i], lw=1, label=lineNames[i])

    ax.set_xlabel(r'$E/\mathrm{keV}$')
    ax.set_ylabel(r'$N$')
    ax.minorticks_on()
    ax.grid(which='minor', visible=True, alpha=0.4)
    ax.grid(which='major', visible=True)
    ax.legend()
    fig.savefig(outFile)
    plt.close(fig)

    return params, paramsErr

def fit_mixing_ratio(unknownFile, elementFiles, plotFile, energyParams, ratioInitialGuess):
    # get E values of the unknown alloy
    n, NUn = load_data(unknownFile)
    # E = linear_fn_odr(energyParams, n)
    samples = len(E)
    # EUn = linear_fn_odr(energyParams, NUn)

    # get N values for the different elements
    count = len(elementFiles)
    NElements = np.zeros((count, samples))
    for i in range(count):
        _, NElements[i] = load_data(elementFiles[i])
        # _, NElement = load_data(elementFile)
        # EElement = linear_fn_odr(energyParams, NElement)
        # EElements.append(EElement)

    # try to find a fitting linear combination of element spectra to imitate the spectrum of the alloy
    def fit_fn(B, x):
        # find index of Energy value closest to x
        xIdx = np.searchsorted(E, x)
        maxMask = xIdx == samples
        xIdx[maxMask] = samples-1
        B = np.array(B)

        result = np.sum(B[:, np.newaxis]*NElements[:, xIdx], 0)
        return result

    def fit_fn_alt(B, x):
        # find index of Energy value closest to x
        xIdxA = np.searchsorted(E, x)
        xIdxB = xIdxA - 1
        maxMask = xIdxA == samples
        xIdxA[maxMask] = samples-1
        maxMask = xIdxB == samples
        xIdxB[maxMask] = samples-1
        minMask = xIdxB == -1
        xIdxB[minMask] = 0
        B = np.array(B)

        result1 = np.sum(B[:, np.newaxis]*NElements[:, xIdxA], 0)
        result2 = np.sum(B[:, np.newaxis]*NElements[:, xIdxB], 0)
        deltaE = E[1]-E[0]
        deltax = x - E[xIdxA]
        result = result1 + deltax/deltaE * (result2-result1)
        return result
    
    params, paramsErr = odr_fit(fit_fn, E, NUn, count, xErr=EErr, p0=ratioInitialGuess)

    fig, ax = plt.subplots()
    ax.plot(E, NUn, label='Messdaten')
    ax.plot(*fit_curve(fit_fn, params, E, 512), label='Anpassung')
    ax.minorticks_on()
    ax.grid(which='minor', visible=True, alpha=0.4)
    ax.grid(which='major', visible=True)
    ax.legend()
    fig.savefig(plotFile)
    plt.close(fig)

    return params, paramsErr

def calc_mass_ratio(rho, eta, etaErr):
    totalMass = np.sum(rho * eta)
    ratios = rho * eta / totalMass
    # ratiosErr = rho * etaErr / np.sum(rho * eta) # TODO: do proper err calculation
    ratiosErr = np.sqrt((rho * etaErr / totalMass)**2 + np.sum((rho**2*eta*etaErr/totalMass**2)**2))

    return ratios, ratiosErr

energyParams, energyParamsErr = fluorescence_energy_calibration(
    'p428/data/5.2/FeZn.txt', 'p428/plot/FeZn_energy_fit.pdf', 'p428/plot/fluorescence_energy_calibration.pdf',
    np.array((6403.84, 7057.98, 8638.86, 9572.0)), # TODO: cite xdb
    p0a=np.array((5400, 104, 4, 2200, 109, 6, 50)),
    p0b=np.array((550, 136, 6, 110, 150, 10, 10))
)
n, _ = load_data('p428/data/5.2/FeZn.txt')
E, EErr = n_to_E(n, energyParams, energyParamsErr)
print('energy params', energyParams, energyParamsErr)
    
inDir = 'p428/data/5.2'
outDir = 'p428/plot/5.2'
# lines = np.array((8047.78, 4510.84, 11442.3, 12613.7)) # Cu, Ti, Au, Pb
lines4 = [(4.510, 8.047, 12.613), (r'Ti $K_\alpha$', r'Cu $K_\alpha$', r'Pb $L_\beta$')]
linesDic = {
    'Unbekannt1': [(5.414, 6.403), (r'Cr $K_\alpha$', r'Fe $K_\alpha$')],
    'Unbekannt2': [(8.047, 8.638), (r'Cu $K_\alpha$', r'Zn $K_\alpha$')],
    # 'Unbekannt3': [(8.047, 8.638, 8.264), (r'Cu $K_\alpha$', r'Zn $K_\alpha$', r'Ni $K_\beta$')],
    'Unbekannt3': [(8.047, 8.638), (r'Cu $K_\alpha$', r'Zn $K_\alpha$')],
    'Unbekannt4': lines4
}

defaultp0 = np.array((2000, 7.5, 2, 0))
p0Fe = np.array((180, 0.6, 0.4, 2200, 6.5, 0.5, 200, 9.5, 3, 0))
fitDic = {
    'Fe': np.array((2200, 6.7, 1.5, 0)), 'Cu': defaultp0, 'Zn': defaultp0, 'Pb': defaultp0, 'Ti': defaultp0
} # 'Fe Cu Zn Pb Ti Cr' # TODO: dont forget Cr
plot_lines_directory(inDir, outDir, energyParams, linesDic=linesDic, fitDic=fitDic)

# determine mass ratio
eta, etaErr = fit_mixing_ratio(
    'p428/data/5.2/Unbekannt4.txt', ('p428/data/5.2/Cu.txt', 'p428/data/5.2/Pb.txt', 'p428/data/5.2/Ti.txt'),
    'p428/plot/Unbekannt4_testfit.pdf', energyParams, (0.347, 0.640, 0.0121)#(0.135, 0.23, 0.1)
)
rho = np.array((8.96, 11.3, 4.51))  # TODO: include source in tex; from https://www.engineersedge.com/materials/densities_of_metals_and_elements_table_13976.htm

ratios, ratiosErr = calc_mass_ratio(rho, eta, etaErr)
print('mass ratios:', ratios, ratiosErr)


# TODO: include statistische Fehler
# TODO: mention that some spectra are way too high
# TODO: mention that errorbars aren't included for visibility


# Metals list: 