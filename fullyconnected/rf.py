import matplotlib
# matplotlib.use('tkagg')
from matplotlib import pyplot as plt
import numpy as np
import sys

layer = 0
weights = []
for arg in sys.argv[1:]:
    tmp = np.load(arg)
    weights.append(tmp['weights'][layer])
    tmp.close()

def plotRF(ws, lim=1.0):
    for i, w in enumerate(ws):
        N1 = int(np.sqrt(w.shape[1]))   # sqrt(784) = 28, you have 28x28 RFs
        N2 = int(np.sqrt(w.shape[0]))   # sqrt(256) = 16, you have 16x16 output cells

        W = np.zeros((N1*N2,N1*N2))             # You are creating a weight wall of 16x16 RF blocks

        for j in range(w.shape[0]):
            r = int(j/N2)
            c = int(j%N2)
            x = c*N1
            y = r*N1
            print W.shape,w.shape
            W[y:y+N1, x:x+N1] = w[j, :].reshape((N1, N1))

        plt.figure(i+1)
        plt.imshow(W, vmin=-lim, vmax=lim)
    plt.show()

if __name__ == '__main__':
    plotRF(weights)
