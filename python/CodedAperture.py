import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hadamard


'''''''''''''''''''''''''''''''''''''''''''''Hadamard Mask'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import HelperFunction as HF
fp = HF.Dialog.SelectFile()
img = HF.FileProcessing.readpng(fp)
HF.FileProcessing.showimg(img)

import CodedAperture as CA
M = CA.HadamardMask(2048, 2048)
M.ApertureFunction()
M.AperturePlot()
S = CA.Shuffled(M.Tx)
S.Shuffle()
S.Tx = S.Tx[int((S.Tx.shape[0] - img.shape[0])/2) : int((S.Tx.shape[0] + img.shape[0])/2), int((S.Tx.shape[1] - img.shape[1])/2) : int((S.Tx.shape[1] + img.shape[1])/2)]
S.AperturePlot()
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


class HadamardMask:

    def __init__(self, n, Y):

        [self.n, self.Y] = [n, Y]

    def ApertureFunction(self):

        self.Tx = (1 - hadamard(self.n)) / 2
        self.Tx = self.Tx[1:, 1:]


    def AperturePlot(self):
        fig = plt.figure()
        ax = fig.subplots()
        cx = ax.imshow(self.Tx, cmap='gist_gray', origin='lower', aspect=1, vmin=0, vmax=1, alpha=1)
        ax.set_title('Hadamard Aperture', fontsize=30)
        cbar = fig.colorbar(cx, ax=ax)
        cbar.ax.tick_params(labelsize=15)



class Shuffled:

    def __init__(self, Tx):

        self.Tx = Tx

    def Shuffle(self):

        rng =np.random.default_rng()
        rng.shuffle(self.Tx, axis=-1)
        rng.shuffle(self.Tx, axis=-2)

    def AperturePlot(self):
        fig = plt.figure()
        ax = fig.subplots()
        cx = ax.imshow(self.Tx, cmap='gist_gray', origin='lower', aspect=1, vmin=0, vmax=1, alpha=1)
        ax.set_title('Shuffled Aperture', fontsize=30)
        cbar = fig.colorbar(cx, ax=ax)
        cbar.ax.tick_params(labelsize=15)