import os
import imageio
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import cmath
import math

os.makedirs("out", exist_ok=True)

sourceImgs = [
    "bn8",
    "bn16",
    "bn32",
    "bn64",
    "bn128",
    "bn256",
]

generatedSizes=[8, 16, 32, 64, 128, 256]

# Generate noise via IDFT
for generatedSize in generatedSizes:
    sourceImg = "idft" + str(generatedSize)
    print(sourceImg)

    # Make random complex numbers that are 
    randomlist = np.empty([generatedSize],dtype=complex)
    for i in range(int(generatedSize/2)):
        randomlist[i] = cmath.rect(np.random.random(1), np.random.random(1)* math.pi * 2)

    for i in range(int(generatedSize/2)-1):
        randomlist[int(generatedSize/2)+1+i] = np.conj(randomlist[int(generatedSize/2)-1-i])

    # TODO: the blue noise real valued signals don't make conjugate results.

    # make a 1d gaussian with a peak where 0hz DC is going to be
    l = generatedSize+1
    sig = 1.9 * float(generatedSize) / 8.0
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    gauss = gauss[:-1].copy()

    # DEBUG: show the gaussian curve
    if False:
        x = np.arange(gauss.shape[0])
        plt.plot(x, gauss)
        plt.show()

    # TODO: gauss needs to be complex. need to multiply both real and imaginary by gauss scalar.

    # make LPF and HPF
    LPF = randomlist * gauss
    HPF = randomlist - LPF

    # Make DC be 1/2 * pixelCount
    LPF[int(generatedSize/2)] = 0.5
    HPF[int(generatedSize/2)] = 0.5
    HPF = HPF * generatedSize
    LPF = LPF * generatedSize
    # TODO: i'm not 100% sure if this is correct, besides the DC term

    # IDFT
    signal = np.fft.ifftn(np.fft.ifftshift(HPF))

    # Normalize the signal
    signal = np.real(signal)
    signalmin = np.amin(signal)
    signalmax = np.amax(signal)
    signal = (signal - signalmin) / (signalmax - signalmin)

    dfty = HPF

    # cosmetic modifications to the DFT    
    #dfty[int(dfty.shape[0] / 2)] = 0 # zero out DC
    dfty = np.append(dfty, dfty[0]) # duplicate the negative dft frequency to the positive

    # Graph the DFT
    plt.title(sourceImg + " DFT") 
    plt.xlabel("Hertz") 
    plt.ylabel("Magnitude")     
    x = np.arange(dfty.shape[0])
    x = x - int((dfty.shape[0]-1) / 2)
    plt.plot(x, dfty)
    fig = plt.gcf()
    fig.savefig("out/" + sourceImg + ".dft.png", bbox_inches='tight')
    plt.close(fig)

    # TODO: clean up all the above!

    # Graph the values
    y = signal
    x = np.arange(y.shape[0])
    plt.plot(x, y)
    ax = plt.gca()
    ax.set_xlim([0.0, y.shape[0]-1])
    ax.set_ylim([0.0, 1.0])
    plt.title(sourceImg + " Values") 
    plt.xlabel("Index") 
    plt.ylabel("Value") 
    fig = plt.gcf()
    fig.savefig("out/" + sourceImg + ".values.png", bbox_inches='tight')
    plt.close(fig)

    # Make Histogram
    plt.title(sourceImg + " Histogram") 
    plt.xlabel("Value") 
    plt.ylabel("Count") 
    plt.hist(y, 256, facecolor='blue', alpha=0.5)
    fig = plt.gcf()
    fig.savefig("out/" + sourceImg + ".histogram.png", bbox_inches='tight')
    plt.close(fig)

# Process blue noise made with void and cluster
for sourceImg in sourceImgs:
    # load the data
    fileName = "source/" + sourceImg + ".png"
    print(fileName)
    y = (imageio.imread(fileName).astype(float) / 255.0)[0,:]

    # Graph the values 
    x = np.arange(y.shape[0])
    plt.plot(x, y)
    ax = plt.gca()
    ax.set_xlim([0.0, y.shape[0]-1])
    ax.set_ylim([0.0, 1.0])
    plt.title(sourceImg + " Values") 
    plt.xlabel("Index") 
    plt.ylabel("Value") 
    fig = plt.gcf()
    fig.savefig("out/" + sourceImg + ".values.png", bbox_inches='tight')
    plt.close(fig)

    # Make Histogram
    plt.title(sourceImg + " Histogram") 
    plt.xlabel("Value") 
    plt.ylabel("Count") 
    plt.hist(y, 256, facecolor='blue', alpha=0.5)
    fig = plt.gcf()
    fig.savefig("out/" + sourceImg + ".histogram.png", bbox_inches='tight')
    plt.close(fig)

    # Make DFT
    dfty = np.abs(np.fft.fftshift(np.fft.fftn(y)))
    #dfty[int(dfty.shape[0] / 2)] = 0 # zero out DC
    dfty = np.append(dfty, dfty[0])

    # TODO: temp
    print(np.fft.fftshift(np.fft.fftn(y)))
    sys.exit()
    
    # Graph the DFT
    plt.title(sourceImg + " DFT") 
    plt.xlabel("Hertz") 
    plt.ylabel("Magnitude")     
    x = np.arange(dfty.shape[0])
    x = x - int((dfty.shape[0]-1) / 2)
    plt.plot(x, dfty)
    fig = plt.gcf()
    fig.savefig("out/" + sourceImg + ".dft.png", bbox_inches='tight')
    plt.close(fig)
    
