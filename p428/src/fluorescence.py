from lib.helpers import *
from pathlib import Path
import matplotlib.pyplot as plt
import lmfit

def Gaussian(x, a, x0, sigma, offset=0):
    """
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
    #(1 / (sigma * np.sqrt(2 * np.pi)))
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

# def multi_gauss_ODR_fit(x, y, nGaussians, xErr=None, yErr=None, centers: np.array=None, heights: np.array=None):
def multi_gauss_ODR_fit(x, y, nGaussians, xErr=None, yErr=None, p0: np.array=None):
    nParams = 3*nGaussians+1
    # p0 = np.ones(nParams)
    # if not (centers is None):
    #     p0[1:-1:3] = centers
    #     p0[0:-1:3] = heights

    return odr_fit(multi_gauss_fn, x, y, nParams, xErr, yErr, p0)

def energy_calibration(dataFilename, plot1Filename, plot2Filename, energyValues, p0a, p0b):
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
    # ax.plot(*fit_curve(multi_gauss_fn, params1, nSlice1, 500), zorder=4)
    # ax.plot(*fit_curve(multi_gauss_fn, params2, nSlice2, 500), zorder=4)
    # for x0i in x0:
    #     ax.axvline(x0i, color='xkcd:gray', label='')
    ax.minorticks_on()
    ax.grid(which='both')
    fig.savefig(plot1Filename)

    fig, ax = plt.subplots()
    ax.errorbar(x0, energyValues, 0, x0Err, 'x', zorder=4)
    ax.errorbar
    ax.plot(*fit_curve(linear_fn_odr, params, x0))
    fig.savefig(plot2Filename)

    return params, paramsErr

def plot_lines_directory(inDir, outDir, energyParams, linesDic):
    """
    Plot all files from inDir and save plots in outDir.
    Use energy calibration with the parameters energyParams and
    plot vertical lines corresponding to the values in linesDic.
    """
    # lines /= 1e3
    colors = ('red', 'green', 'blue', 'yellow', 'purple')

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
        # for line in lines:
        #     ax.axvline(line)
        if sampleName in linesDic:
            lines, lineNames = linesDic[sampleName]
            for i in range(len(lines)):
                ax.axvline(lines[i], color=colors[i], lw=1, label=lineNames[i])
        ax.set_xlabel(r'$E/\mathrm{keV}$')
        ax.set_ylabel(r'$N$')
        ax.legend()
        ax.minorticks_on()
        ax.grid(which='minor', alpha=0.5)
        ax.grid(which='major')
        outFilename = str(Path(outDir))+'/'+sampleName+'.pdf'
        fig.savefig(outFilename)


params, paramsErr = energy_calibration(
    'p428/data/5.2/FeZn.txt', 'p428/plot/FeZn_energy_fit.pdf', 'p428/plot/energy_calibration.pdf',
    np.array((6403.84, 7057.98, 8638.86, 9572.0)), # TODO: cite xdb
    p0a=np.array((5400, 104, 4, 2200, 109, 6, 50)),
    p0b=np.array((550, 136, 6, 110, 150, 10, 10)))
print(params)
    
inDir = 'p428/data/5.2'
outDir = 'p428/plot/5.2'
lines = np.array((8047.78, 4510.84, 11442.3, 12613.7)) # Cu, Ti, Au, Pb
linesDic = {
    'Unbekannt1': [(5.414, 6.403), (r'Cr $K_\alpha$', r'Fe $K_\alpha$')],
    'Unbekannt2': [(8.047, 8.638), (r'Cu $K_\alpha$', r'Zn $K_\alpha$')],
    'Unbekannt3': [(8.047, 8.638, 8.264), (r'Cu $K_\alpha$', r'Zn $K_\alpha$', r'Ni $K_\beta$')],
    'Unbekannt4': [(4.510, 8.047, 12.613), (r'Ti $K_\alpha$', r'Cu $K_\alpha$', r'Pb $L_\beta$')]
}
plot_lines_directory(inDir, outDir, params, linesDic)
# energy_calibration('p428/data/5.2/FeZn.txt', 'p428/plot/FeZn_raw.pdf', np.array((104, 112, 137, 155)), np.array((5000, 2000, 600, 100)))