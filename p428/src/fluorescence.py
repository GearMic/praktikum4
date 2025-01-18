from lib.helpers import *
from pathlib import Path
import matplotlib.pyplot as plt
import lmfit

pltColors = ('red', 'green', 'blue', 'yellow', 'purple')

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

def multi_gauss_ODR_fit(x, y, nGaussians, xErr=None, yErr=None, p0: np.array=None):
    nParams = 3*nGaussians+1
    # p0 = np.ones(nParams)
    # if not (centers is None):
    #     p0[1:-1:3] = centers
    #     p0[0:-1:3] = heights

    return odr_fit(multi_gauss_fn, x, y, nParams, xErr, yErr, p0)

def fluorescence_energy_calibration(dataFilename, plot1Filename, plot2Filename, energyValues, p0a, p0b):
    n, N = load_data(dataFilename)

    # fit gaussians to data
    nSlice, NSlice = slice_from_range(80, 160, n, N)
    paramsGauss, paramsGaussErr = multi_gauss_ODR_fit(nSlice, NSlice, 4, p0=np.concatenate((p0a[:-1], p0b)))

    # do linear fit for energy calibration
    energyValues /= 1e3
    x0 = paramsGauss[1:-1:3]
    x0Err = paramsGaussErr[1:-1:3] # TODO: is the slice done by reference?
    params, paramsErr = odr_fit(linear_fn_odr, x0, energyValues, 2, xErr=x0Err)

    fig, ax = plt.subplots()
    ax.set_xlim(80, 160)
    ax.plot(n, N)
    ax.plot(*fit_curve(multi_gauss_fn, paramsGauss, nSlice, 500), zorder=4)
    ax.minorticks_on()
    ax.grid(which='both')
    fig.savefig(plot1Filename)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.errorbar(x0, energyValues, 0, x0Err, 'x', zorder=4)
    ax.errorbar
    ax.plot(*fit_curve(linear_fn_odr, params, x0))
    fig.savefig(plot2Filename)
    plt.close(fig)

    return params, paramsErr

def plot_lines_directory(inDir, outDir, energyParams, linesDic):
    """
    Plot all files from inDir and save plots in outDir.
    Use energy calibration with the parameters energyParams and
    plot vertical lines corresponding to the values in linesDic.
    """

    inDir = Path(inDir)
    for file in inDir.iterdir():
        if not file.is_file():
            continue
        sampleName = file.stem
        
        n, N = load_data(file)
        Nerr = 2
        E = linear_fn_odr(energyParams, n)

        fig, ax = plt.subplots()
        ax.plot(E, N, label='Messdaten')
        if sampleName in linesDic:
            lines, lineNames = linesDic[sampleName]
            for i in range(len(lines)):
                ax.axvline(lines[i], color=pltColors[i], lw=1, label=lineNames[i])
        ax.set_xlabel(r'$E/\mathrm{keV}$')
        ax.set_ylabel(r'$N$')
        ax.legend()
        ax.minorticks_on()
        ax.grid(which='minor', visible=True, alpha=0.5)
        ax.grid(which='major', visible=True)
        outFilename = str(Path(outDir))+'/'+sampleName+'.pdf'
        fig.savefig(outFilename)
        plt.close(fig)

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


energyParams, energyParamsErr = fluorescence_energy_calibration(
    'p428/data/5.2/FeZn.txt', 'p428/plot/FeZn_energy_fit.pdf', 'p428/plot/fluorescence_energy_calibration.pdf',
    np.array((6403.84, 7057.98, 8638.86, 9572.0)), # TODO: cite xdb
    p0a=np.array((5400, 104, 4, 2200, 109, 6, 50)),
    p0b=np.array((550, 136, 6, 110, 150, 10, 10)))
    
inDir = 'p428/data/5.2'
outDir = 'p428/plot/5.2'
# lines = np.array((8047.78, 4510.84, 11442.3, 12613.7)) # Cu, Ti, Au, Pb
lines4 = [(4.510, 8.047, 12.613), (r'Ti $K_\alpha$', r'Cu $K_\alpha$', r'Pb $L_\beta$')]
linesDic = {
    'Unbekannt1': [(5.414, 6.403), (r'Cr $K_\alpha$', r'Fe $K_\alpha$')],
    'Unbekannt2': [(8.047, 8.638), (r'Cu $K_\alpha$', r'Zn $K_\alpha$')],
    'Unbekannt3': [(8.047, 8.638, 8.264), (r'Cu $K_\alpha$', r'Zn $K_\alpha$', r'Ni $K_\beta$')],
    'Unbekannt4': lines4
}
plot_lines_directory(inDir, outDir, energyParams, linesDic)

params4, params4Err = fit_fluorescence_data(
    'p428/data/5.2/Unbekannt4.txt', 'p428/plot/Unbekannt4_fit.pdf',
    energyParams, lines4,
    3, p0=np.array((60, 4.510, 0.5, 270, 8.05, 0.5, 30, 11.0, 0.3, 1))
)
print(params4)
# TODO: include statistische Fehler