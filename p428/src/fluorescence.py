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
def energy_calibration(inFilename, outFilename, energyValues, p0a, p0b):
    n, N = load_data(inFilename)

    # nSlice1, NSlice1 = slice_from_range(80, 160, n, N)
    # params1, params1Err = multi_gauss_ODR_fit(nSlice1, NSlice1, 4, p0=np.concatenate((p0a[:-1], p0b)))

    nSlice1, NSlice1 = slice_from_range(80, 122, n, N)
    params1, params1Err = multi_gauss_ODR_fit(nSlice1, NSlice1, 2, p0=p0a)
    nSlice2, NSlice2 = slice_from_range(126, 160, n, N)
    params2, params2Err = multi_gauss_ODR_fit(nSlice2, NSlice2, 2, p0=p0b)
    print(params1)
    print(params2)

    fig, ax = plt.subplots()
    ax.set_xlim(80, 160)
    ax.plot(n, N)
    ax.plot(*fit_curve(multi_gauss_fn, params1, nSlice1, 500), zorder=4)
    ax.plot(*fit_curve(multi_gauss_fn, params2, nSlice2, 500), zorder=4)
    ax.minorticks_on()
    ax.grid(which='both')
    fig.savefig(outFilename)

    a, b = 0, 0
    return a, b

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
energy_calibration(
    'p428/data/5.2/FeZn.txt', 'p428/plot/FeZn_raw3.pdf',
    (6403.84, 7057.98, 8638.86, 9572.0), # TODO: cite xdb
    p0a=np.array((5400, 104, 4, 2200, 109, 6, 50)),
    p0b=np.array((550, 136, 6, 110, 150, 10, 10)))
    # p0=np.array((5000, 100, 4, 1000, 112, 3, 600, 136, 3, 100, 150, 10, 50)))


""" 

def plot_data_fit(fig, ax, alpha, y, yErr, params=None):
    ax.errorbar(alpha, y, yErr, fmt='-', label='Daten', lw=0.5)
    # ax.plot(alpha, y, '-', lw=1, label='Daten')

    # paramsPrint = np.array((params, paramsErr)).T
    # paramsPrint = (r' \pm '.join(tuple(np.array(param, dtype=str))) for param in paramsPrint)
    # paramsPrint = r',\ '.join(paramsPrint)
    # print(paramsPrint)

    if not (params is None):
        xFit = array_range(alpha, overhang=0)
        yFit = double_gauss_fn(xFit, *params)
        ax.plot(xFit, yFit, label='Kalibrierungskurve')

def full_gauss_fit_for_lines():
    omegaG = np.array((13.8, 18.2, 37.0))
    alphaRange = np.array(((-0.5, 0.9), (-0.06, 0.04), (-0.2, 0.1)))
    p0 = ((20, -0.02, 0.05, 20, 0, 0.05, 40), (10, -0.03, 0.01, 20, 0, 0.02, 5), (20, -0.1, 0.02, 80, 0, 0.02, 5))
    doFit = (False, True, True)
    inFilenames = tuple('p402/data/ccd/line%.1f.txt' % omega for omega in omegaG)
    outFilenames = tuple('p402/plot/line%.1f.pdf' % omega for omega in omegaG)
    lowerBounds = np.array((0, -np.inf, 0, 0, -np.inf, 0, 0))
    upperBounds = np.array((np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf))

    params, paramsErr = np.zeros((len(omegaG), 7)), np.zeros((len(omegaG), 7))
    for i in range(len(inFilenames)):
    # for i in range(9, 10):
        lower, upper = alphaRange[i, 0], alphaRange[i, 1]

        alpha, y, yErr = load_data(inFilenames[i], 0.0001, 0.5)
        rangeMask = (alpha >= np.full_like(alpha, lower)) & (alpha <= np.full_like(alpha, upper))
        alpha, y, yErr = alpha[rangeMask], y[rangeMask], yErr[rangeMask]

        # fit data
        param, paramErr = None, None
        if doFit[i]:
            param, paramErr = chisq_fit(
                double_gauss_fn, alpha, y, yErr, p0=p0[i],
                bounds = (lowerBounds, upperBounds), absolute_sigma=True)

            params[i] = param
            paramsErr[i] = paramErr
        

        fig, ax = plt.subplots()

        plot_data_fit(fig, ax, alpha, y, yErr, param)
        ax.legend()
        ax.minorticks_on()
        ax.grid(which='both')

        # ax.set_title('Linien bei $\\omega_G=%.1f°$' % omegaG[i])
        ax.set_xlabel(r'Position $\gamma$/°')
        ax.set_ylabel(r'Intensität $I$/%')
        fig.savefig(outFilenames[i])

    mu1, mu1Err = params[:, 1], paramsErr[:, 1]
    mu2, mu2Err = params[:, 4], paramsErr[:, 4]
    deltaBeta = np.abs(mu2 - mu1)
    deltaBetaErr = np.sqrt(mu1Err**2 + mu2Err**2)

    omegaB = 140
    beta = (180+omegaG-omegaB)
    betaErr = 0.6
    alpha = omegaG

    paramsFrame = pd.DataFrame({
        r'$\alpha/°$': alpha, r'$\beta/°$': beta,
        r'$\mu_1/°$': mu1, r'$\Delta\mu_1/°$': mu1Err, r'$\mu_2/°$': mu2, r'$\Delta\mu_2/°$': mu2Err,
        r'$\delta\beta/°$': deltaBeta, r'$\Delta\delta\beta/°$': deltaBetaErr, 
    })
    paramsFrame.to_csv('p402/data/balmer_gauss_fit.csv', index=False)

    # return alpha, beta, deltaBeta, betaErr, deltaBetaErr, paramsFrame
    return np.deg2rad(alpha), np.deg2rad(beta), np.deg2rad(deltaBeta), np.deg2rad(betaErr), np.deg2rad(deltaBetaErr), paramsFrame

 """