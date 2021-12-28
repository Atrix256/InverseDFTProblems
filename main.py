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

for sourceImg in sourceImgs:
    fileName = "source/" + sourceImg + ".png"
    y = (imageio.imread(fileName).astype(float) / 255.0)[0,:]
    x = np.arange(y.shape[0])
    plt.plot(x, y)
    ax = plt.gca()
    ax.set_xlim([0.0, y.shape[0]-1])
    ax.set_ylim([0.0, 1.0])
    plt.title(sourceImg + " Values") 
    plt.xlabel("Index") 
    plt.ylabel("Value") 
    plt.show()
