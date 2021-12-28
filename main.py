import os
import imageio
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("out", exist_ok=True)

sourceImgs = [
    "bn8",
    "bn16",
    "bn32",
    "bn64",
    "bn128",
    "bn256",
]

# Process blue noise made with void and cluster
for sourceImg in sourceImgs:
    # load the data
    fileName = "source/" + sourceImg + ".png"
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
    
