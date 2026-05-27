import numpy as np
from scipy.optimize import nnls
from scipy.ndimage import shift
import matplotlib.pyplot as plt
import cv2


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

import DispersivePass as DP
ApertureConst = 4
P = DP.Dispersion(S.Tx, ApertureConst, img.shape[2])
DispersionPTN = P.CalcDispersionPattern(P.ApplySourceCube(img))
P.Plotting(DispersionPTN, 'Coded Aperture Simulation')
P.Plotting(img, 'Incident Cube')

import ReconstructionSpectrum as RS
R = RS.NonNegativeLeastSquare(S.Tx, P.ApertureConst, P.NumChannel, DispersionPTN)
R.Reconstruction_NNLS()
R.Plotting(R.X, 'Reconstructed Spectrum')
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


class NonNegativeLeastSquare:

    def __init__(self, Aperture, ApertureConst, NumChannel, y):

        self.Aperture, self.NumChannel, self.ApertureConst, self.y = Aperture, NumChannel, ApertureConst, y
        self.Height, self.Width = self.Aperture.shape
        self.CameraRes = (self.Height, self.Width + self.ApertureConst*(self.NumChannel - 1))

    def Reconstruction_NNLS(self):
        self.X = self._Reconstruction_by_NNLS(self.y, self.Height, self.Width, self.CameraRes, self.NumChannel, self.Aperture, self.ApertureConst)

    def _Reconstruction_by_NNLS(self, y, height, width, CamRes, NumCh, Aperture, ApertureConst):
        X = np.zeros((height, width, NumCh))
        for k in range(height):
            A_row = Aperture[k]
            H_row = self.__MakeTransferFunction(CamRes, NumCh, A_row, ApertureConst, width)
            y_row = y[k]

            x_row_flat = self.__CalcNNLS(H_row, y_row)
            x_row = x_row_flat.reshape(NumCh, width).T
            X[k] = x_row
            print (f'{100 * k / height} %')

        return X

    def __MakeTransferFunction(self, CamRes, ch, Aperture_row, ApertureConst, Width):

        H_row = np.zeros((CamRes[1], Width * ch ))
        for k in range(ch):
            H_row[k * ApertureConst : Width + k * ApertureConst, k * Width : (k+1) * Width] = Aperture_row
        return H_row

    def __CalcNNLS(self, A, b):
        return np.array(nnls(A, b)[0])


    def Plotting(self, x, title):

        fig = plt.figure()
        ax = fig.subplots()
        cx = ax.imshow((np.abs(x)), cmap='gist_gray', origin='lower', vmin=0, alpha=1)
        self.forceAspect(ax)
        ax.set_title(title, fontsize=30)
        cbar = fig.colorbar(cx, ax=ax)
        cbar.ax.tick_params(labelsize=15)


    def forceAspect(self, ax, aspect=1):
        im = ax.get_images()
        extent = im[0].get_extent()
        ax.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect)