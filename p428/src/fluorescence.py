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

# def energy_calibration(inFilename, outFilename, centers=None, heights=None):
# def energy_calibration(inFilename, outFilename, energyValues, **kwargs):
def energy_calibration(dataFilename, plot1Filename, plot2Filename, energyValues, p0a, p0b):
    n, N = load_data(dataFilename)

    nSlice, NSlice = slice_from_range(80, 160, n, N)
    params, paramsErr = multi_gauss_ODR_fit(nSlice, NSlice, 4, p0=np.concatenate((p0a[:-1], p0b)))

    # nSlice1, NSlice1 = slice_from_range(80, 122, n, N)
    # params1, params1Err = multi_gauss_ODR_fit(nSlice1, NSlice1, 2, p0=p0a)
    # nSlice2, NSlice2 = slice_from_range(126, 160, n, N)
    # params2, params2Err = multi_gauss_ODR_fit(nSlice2, NSlice2, 2, p0=p0b)
    print(params)
    # print(params2)

    fig, ax = plt.subplots()
    ax.set_xlim(80, 160)
    ax.plot(n, N)
    ax.plot(*fit_curve(multi_gauss_fn, params, nSlice, 500), zorder=4)
    # ax.plot(*fit_curve(multi_gauss_fn, params1, nSlice1, 500), zorder=4)
    # ax.plot(*fit_curve(multi_gauss_fn, params2, nSlice2, 500), zorder=4)
    ax.minorticks_on()
    ax.grid(which='both')
    fig.savefig(plot1Filename)

    # do linear fit for energy calibration
    energyValues /= 1e3
    x0 = params[1:-1:3]
    x0Err = paramsErr[1:-1:3] # TODO: is the slice done by reference?
    params, paramsErr = odr_fit(linear_fn_odr, x0, energyValues, 2, xErr=paramsErr)

    fig, ax = plt.subplots()
    ax.errorbar(x0, energyValues, 0, x0Err, 'x', zorder=4)
    ax.errorbar
    ax.plot(*fit_curve(linear_fn_odr, params, x0))
    fig.savefig(plot2Filename)

    return params, paramsErr

def directory_gauss_fit(inDir, outDir):
    """Do Gauss Fits on an entire directory of Data."""

    inDir = Path(inDir)
    for file in inDir.iterdir():
        if not file.is_file():
            continue
        
        n, N = load_data(file)
        Nerr = 2

        fig, ax = plt.subplots()
        ax.plot(n, N)
        ax.minorticks_on()
        ax.grid(which='both')

        outFilename = str(Path(outDir))+'/'+file.stem+'.pdf'
        fig.savefig(outFilename)

    
inDir = 'p428/data/5.2'
outDir = 'p428/plot/5.2'
# directory_gauss_fit(inDir, outDir)
# energy_calibration('p428/data/5.2/FeZn.txt', 'p428/plot/FeZn_raw.pdf', np.array((104, 112, 137, 155)), np.array((5000, 2000, 600, 100)))
params, paramsErr = energy_calibration(
    'p428/data/5.2/FeZn.txt', 'p428/plot/FeZn_energy_fit.pdf', 'p428/plot/energy_calibration.pdf',
    np.array((6403.84, 7057.98, 8638.86, 9572.0)), # TODO: cite xdb
    p0a=np.array((5400, 104, 4, 2200, 109, 6, 50)),
    p0b=np.array((550, 136, 6, 110, 150, 10, 10)))
    # p0=np.array((5000, 100, 4, 1000, 112, 3, 600, 136, 3, 100, 150, 10, 50)))

print(params)

