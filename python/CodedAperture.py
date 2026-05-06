import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hadamard

'''''''''''''''''''''''''''''''''''''''''''''Harmonic Mask'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# import CodedAperture as CA
# M = CA.HarmonicMask(64, 90)
# M.ApertureFunction()
# M.AperturePlot()
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

'''''''''''''''''''''''''''''''''''''''''''''Hadamard Mask'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# import CodedAperture as CA
# M = CA.HadamardMask(64, 90)
# M.ApertureFunction()
# M.AperturePlot()
# S = CA.Shuffled(M.Tx)
# S.Shuffle()
# S.AperturePlot()
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

'''''''''''''''''''''''''''''''''''''''''Row Doubled Hadamard Mask''''''''''''''''''''''''''''''''''''''''''''''''''''''
# import CodedAperture as CA
# M = CA.RowDoubledHadamardMask(64, 90)
# M.ApertureFunction()
# M.AperturePlot()
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


class HarmonicMask:

    def __init__(self, m, Y):

        [self.m, self.Y] = [m, Y]

        self.marr = np.linspace(1, self.m, self.m)
        self.Yarr = np.linspace(self.Y, -self.Y, 2*self.Y)
        self.mv, self.Yv = np.meshgrid(self.marr, self.Yarr)
        self.Comp = self.mv * np.pi * self.Yv / self.Y

    def ApertureFunction(self):

        self.Tx = (1 + np.cos(self.Comp)) / 2

    def AperturePlot(self):

        fig = plt.figure()
        ax = fig.subplots()
        cx = ax.imshow(self.Tx, cmap='gist_gray', origin='lower', aspect=1, vmin=0, vmax=1, alpha=1)
        ax.set_title('Harmonic Aperture', fontsize=30)
        cbar = fig.colorbar(cx, ax=ax)
        cbar.ax.tick_params(labelsize=15)


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



class RowDoubledHadamardMask:

    def __init__(self, n, Y):

        [self.n, self.Y] = [n, Y]
        self.Tx = np.zeros((0, self.n))

    def ApertureFunction(self):

        Tx1 = (1 + hadamard(self.n)) / 2
        Tx2 = (1 - hadamard(self.n)) / 2

        for k in range(self.n):
            self.Tx = np.vstack([self.Tx[:, :], Tx1[k, :]])
            self.Tx = np.vstack([self.Tx[:, :], Tx2[k, :]])

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

        np.random.shuffle(self.Tx)

    def AperturePlot(self):
        fig = plt.figure()
        ax = fig.subplots()
        cx = ax.imshow(self.Tx, cmap='gist_gray', origin='lower', aspect=1, vmin=0, vmax=1, alpha=1)
        ax.set_title('Shuffled Aperture', fontsize=30)
        cbar = fig.colorbar(cx, ax=ax)
        cbar.ax.tick_params(labelsize=15)