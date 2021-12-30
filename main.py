import os
import imageio
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import cmath
import math
from PIL import Image, ImageFont, ImageDraw 

os.makedirs("out", exist_ok=True)

sourceImgs = [
    "bn8",
    "bn16",
    "bn32",
    "bn64",
    "bn128",
    "bn256",
    "bn10240",
]

generatedSizes=[8, 16, 32, 64, 128, 256, 10240]

# Generate noise via IDFT
for generatedSize in generatedSizes:
    sourceImg = "idft" + str(generatedSize)
    print(sourceImg)

    # Make random complex numbers that are 
    randomlist = np.empty([generatedSize],dtype=complex)
    for i in range(int(generatedSize/2)):
        angle = 0.0
        if i > 0:
            angle = np.random.random(1)* math.pi * 2
        randomlist[i] = cmath.rect(np.random.random(1), angle)

    for i in range(int(generatedSize/2)-1):
        randomlist[int(generatedSize/2)+1+i] = np.conj(randomlist[int(generatedSize/2)-1-i])

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

    # make LPF and HPF    
    LPF = randomlist * gauss
    HPF = randomlist - LPF

    # Make DC be 1/2 * pixelCount
    HPF = HPF * generatedSize
    LPF = LPF * generatedSize
    LPF[int(generatedSize/2)] = (generatedSize-1)/2
    HPF[int(generatedSize/2)] = (generatedSize-1)/2

    DFTs = [LPF, HPF]
    labels = ["LPF", "HPF"]

    for dft, label in zip(DFTs, labels):

        sourceImg = "idft" + str(generatedSize) + "_" + label
        print(label)
        
        # IDFT
        signal = np.fft.ifftn(np.fft.ifftshift(dft))

        # DEBUG: make sure the imaginary part of the signal isn't significant
        if False:
            for signalSample in signal:
                if np.imag(signalSample) > 0.1:
                    print("===============")
                    print("randomlist = " + str(randomlist))
                    print("gauss = " + str(gauss))
                    print("dft = " + str(dft))
                    print("signal = " + str(signal))
                    sys.exit()

        dfty = np.abs(dft)

        # cosmetic modifications to the DFT    
        dfty[int(dfty.shape[0] / 2)] = 0 # zero out DC
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

        # Graph the raw values
        signal = np.real(signal)
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
        fig.savefig("out/" + sourceImg + ".valuesraw.png", bbox_inches='tight')
        plt.close(fig)

        # Make Raw Histogram
        if False:
            plt.title(sourceImg + " Histogram") 
            plt.xlabel("Value") 
            plt.ylabel("Count") 
            plt.hist(y, 256, facecolor='blue', alpha=0.5)
            fig = plt.gcf()
            fig.savefig("out/" + sourceImg + ".histogramraw.png", bbox_inches='tight')
            plt.close(fig)

        # DEBUG: verify dft
        if False:
            # Make DFT
            dfty = np.abs(np.fft.fftshift(np.fft.fftn(signal)))
            dfty[int(dfty.shape[0] / 2)] = 0 # zero out DC
            dfty = np.append(dfty, dfty[0])
            
            # Graph the DFT
            plt.title(sourceImg + " DFT") 
            plt.xlabel("Hertz") 
            plt.ylabel("Magnitude")     
            x = np.arange(dfty.shape[0])
            x = x - int((dfty.shape[0]-1) / 2)
            plt.plot(x, dfty)
            fig = plt.gcf()
            fig.savefig("out/" + sourceImg + ".dft2raw.png", bbox_inches='tight')
            plt.close(fig)    

        # Normalize the signal
        signalmin = np.amin(signal)
        signalmax = np.amax(signal)
        signal = (signal - signalmin) / (signalmax - signalmin)

        # Graph the values
        signal = np.real(signal)
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
        if True:
            plt.title(sourceImg + " Histogram") 
            plt.xlabel("Value") 
            plt.ylabel("Count") 
            plt.hist(y, 256, facecolor='blue', alpha=0.5)
            fig = plt.gcf()
            fig.savefig("out/" + sourceImg + ".histogram.png", bbox_inches='tight')
            plt.close(fig)

        # DEBUG: verify dft
        if False:
            # Make DFT
            dfty = np.abs(np.fft.fftshift(np.fft.fftn(signal)))
            dfty[int(dfty.shape[0] / 2)] = 0 # zero out DC
            dfty = np.append(dfty, dfty[0])
            
            # Graph the DFT
            plt.title(sourceImg + " DFT") 
            plt.xlabel("Hertz") 
            plt.ylabel("Magnitude")     
            x = np.arange(dfty.shape[0])
            x = x - int((dfty.shape[0]-1) / 2)
            plt.plot(x, dfty)
            fig = plt.gcf()
            fig.savefig("out/" + sourceImg + ".dft2.png", bbox_inches='tight')
            plt.close(fig)

        # tile the signal to be a 256x256 image
        if generatedSize <= 256:
            row = Image.new('L', (1, 256), (255))
            signalImage = Image.fromarray(np.uint8(signal*255))
            for i in range(int(256/generatedSize)):
                row.paste(signalImage, (0, i * generatedSize))
            out = Image.new('L', (256, 256), (255))
            for i in range(256):
                out.paste(row, (i, 0))
            out.save("out/" + sourceImg + ".png")

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
    dfty[int(dfty.shape[0] / 2)] = 0 # zero out DC
    dfty = np.append(dfty, dfty[0])

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

    # tile the signal to be a 256x256 image
    generatedSize = y.shape[0]
    if generatedSize <= 256:
        row = Image.new('L', (1, 256), (255))
        signalImage = Image.fromarray(np.uint8(y*255))
        for i in range(int(256/generatedSize)):
            row.paste(signalImage, (0, i * generatedSize))
        out = Image.new('L', (256, 256), (255))
        for i in range(256):
            out.paste(row, (i, 0))
        out.save("out/" + sourceImg + ".png")
