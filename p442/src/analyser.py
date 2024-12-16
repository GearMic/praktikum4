from helpers import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_data(filename):
    rawData = read_data_pd(filename)
    nr = np.array(rawData['nr.'])
    x = np.array(rawData['x/skt'])

    return nr, x

def preprocess_data(leftIndex, rightIndex):
    # left, right = np.min(x), np.max(x)
    # left, right = x[1], x[4]
    left, right = x[leftIndex], x[rightIndex]
    fsr = right-left
    fsrErr = np.sqrt(2)*xErr

    ratio = (x-left)/fsr
    ratioErr = np.sqrt(2*(xErr/fsr)**2 + ((x-left)*fsrErr/fsr**2)**2)
    deltaNu = ratio*modeSep
    deltaNuErr = ratioErr*modeSep

    print('fsr:', fsr, fsrErr)
    print('ratio', ratio)
    print('ratioErr', ratioErr)
    print('deltaNu/GHz', deltaNu/1e9)
    print('deltaNuErr/GHz', deltaNuErr/1e9)

    dataFrame = pd.DataFrame({
        r'Nr.': nr,
        r'$x/\mathrm{Skt.}$': x,
        r'$\delta\nu/\mathrm{FSR}$': ratio,
        r'$\Delta\delta\nu/\mathrm{FSR}$': ratioErr,
        # r'$\delta\nu/\si\MHz$': deltaNu/1e6,
        # r'$\Delta\delta\nu/\si\MHz$': deltaNuErr/1e6
        r'$\delta\nu$': deltaNu/1e6,
        r'$\Delta\delta\nu$': deltaNuErr/1e6
    })

    # return ratio, deltaNu, fsr, ratioErr, deltaNuErr, fsrErr
    return dataFrame

c = 2.998e8
l = 5e-2
modeSep = c/l/4
xErr = 0.3

## short resonator
nr, x = load_data('p442/data/5.7analyser_short.csv')
dataFrame = preprocess_data(1, 4)
dataFrame.to_latex('p442/data/5.7analyser_short.auto.tex', index=False)
# TODO: calculate ratio to laser FSR

## long resonator
nr, x = load_data('p442/data/5.7analyser_long.csv')
dataFrame = preprocess_data(0, -1)
dataFrame.to_latex('p442/data/5.7analyser_long.auto.tex', index=False)