import numpy as np
from scipy.optimize import nnls
from scipy.ndimage import shift
import matplotlib.pyplot as plt
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr

import HelperFunction as HF


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
ApertureConst = 500
P = DP.Dispersion(S.Tx, ApertureConst, img.shape[2])
DispersionPTN = P.CalcDispersionPattern(P.ApplySourceCube(img))
HF.Dialog.Plotting(DispersionPTN, 'Coded Aperture Simulation')
HF.Dialog.Plotting(img, 'Incident Cube')

import ReconstructionSpectrum as RS
R = RS.NonNegativeLeastSquare(S.Tx, P.ApertureConst, P.NumChannel, DispersionPTN)
R.Reconstruction_NNLS()
HF.Dialog.Plotting(R.X, 'Reconstructed Spectrum')
from skimage.metrics import peak_signal_noise_ratio as psnr
print(f"PSNR: {psnr(img, R.X, data_range = 255)}")
print(f"PSNR: {HF.DataProcessing.CalcPSNR(img, R.X, data_range = 255)}")

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

'''''''''''''''''''''''''''''''''''''''''''''MultiShot Hadamard Mask'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import HelperFunction as HF
fp = HF.Dialog.SelectFile()

img = HF.FileProcessing.readpng(fp)
HF.FileProcessing.showimg(img)

import CodedAperture as CA
import numpy as np
M = CA.HadamardMask(2048, 2048)
M.ApertureFunction()
M.AperturePlot()
S = CA.Shuffled(M.Tx)
S.Shuffle()
S.Tx = S.Tx[int((S.Tx.shape[0] - img.shape[0])/2) : int((S.Tx.shape[0] + img.shape[0])/2), int((S.Tx.shape[1] - img.shape[1])/2) : int((S.Tx.shape[1] + img.shape[1])/2)]
S.AperturePlot()
S1 = S.Tx.copy()
S2 = np.roll(S.Tx, shift = 1, axis = 1)
S3 = np.roll(S.Tx, shift = 1, axis = 0)
S4 = np.roll(S.Tx, shift = [1, 1], axis = [0, 1])
ApertureList = [S1, S2, S3, S4]

import DispersivePass as DP
ApertureConst = 500
P = DP.MultiShotDispersion(ApertureList, ApertureConst, img.shape[2])
DispersionPTNs = P.CalcDispersionPattern(P.ApplySourceCube(img))
HF.Dialog.Plotting(DispersionPTNs[0], 'Coded Aperture Simulation')
HF.Dialog.Plotting(img, 'Incident Cube')

import ReconstructionSpectrum as RS
R = RS.MultiShotNonNegativeLeastSquare(ApertureList, P.ApertureConst, P.NumChannel, DispersionPTNs)
R.Reconstruction_NNLS()
HF.Dialog.Plotting(R.X, 'Reconstructed Spectrum')
from skimage.metrics import peak_signal_noise_ratio as psnr
print(f"PSNR: {psnr(img, R.X, data_range = 255)}")
print(f"PSNR: {HF.DataProcessing.CalcPSNR(img, R.X, data_range = 255)}")

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
            H_row[k * ApertureConst : Width + k * ApertureConst, k * Width : (k+1) * Width] = np.diag(Aperture_row)
        return H_row

    def __CalcNNLS(self, A, b):
        return np.array(nnls(A, b)[0])


class MultiShotNonNegativeLeastSquare:

    def __init__(self, ApertureList, ApertureConst, NumChannel, y_list):

        self.ApertureList, self.NumChannel, self.ApertureConst, self.y_list = ApertureList, NumChannel, ApertureConst, y_list
        self.NumShots = len(ApertureList)
        self.Height, self.Width = self.ApertureList[0].shape
        self.CameraRes = (self.Height, self.Width + self.ApertureConst*(self.NumChannel - 1))

    def Reconstruction_NNLS(self):
        self.X = self._Reconstruction_by_NNLS(self.y_list, self.Height, self.Width, self.CameraRes, self.NumChannel, self.ApertureList, self.NumShots, self.ApertureConst)

    def _Reconstruction_by_NNLS(self, y_list, height, width, CamRes, NumCh, ApertureList, NumShots, ApertureConst):
        X = np.zeros((height, width, NumCh))
        for k in range(height):
            H_row_stacked = []
            y_row_stacked = []

            for shot_now in range(self.NumShots):

                A_row = ApertureList[shot_now][k]
                H_row_stacked.append(self.__MakeTransferFunction(CamRes, NumCh, A_row, ApertureConst, width))
                y_row_stacked.append(y_list[shot_now][k])

            H_row = np.vstack(H_row_stacked)
            y_row = np.concatenate(y_row_stacked)

            x_row_flat = self.__CalcNNLS(H_row, y_row)
            x_row = x_row_flat.reshape(NumCh, width).T
            X[k] = x_row
            print (f'{100 * k / height} %')

        return X

    def __MakeTransferFunction(self, CamRes, ch, Aperture_row, ApertureConst, Width):

        H_row = np.zeros((CamRes[1], Width * ch ))
        for k in range(ch):
            H_row[k * ApertureConst : Width + k * ApertureConst, k * Width : (k+1) * Width] = np.diag(Aperture_row)
        return H_row

    def __CalcNNLS(self, A, b):
        return np.array(nnls(A, b)[0])