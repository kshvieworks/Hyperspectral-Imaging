import numpy as np
import matplotlib.pyplot as plt


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
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

'''''''''''''''''''''''''''''''''''''''''Row Doubled Hadamard Mask''''''''''''''''''''''''''''''''''''''''''''''''''''''
# import CodedAperture as CA
# M = CA.RowDoubledHadamardMask(64, 64)
# M.ApertureFunction()
# M.AperturePlot()
# import DispersivePass as DP
# P = DP.Dispersion(M.Tx, 4, 2000)
# P.Plotting()
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


class Dispersion:

    def __init__(self, Aperture, ApertureConst, NumChannel):
        [self.Aperture, self.ApertureConst, self.NumChannel] = [Aperture, ApertureConst, NumChannel]

        self.ApertureHeight, self.ApertureWidth = self.Aperture.shape
        self.CameraRes = (self.ApertureHeight, self.ApertureWidth + self.ApertureConst*(self.NumChannel - 1))

    def CalcDispersionPattern(self, MaskedScene):

        temp = np.zeros(self.CameraRes)
        for k in range(self.NumChannel):
            temp[:, k * self.ApertureConst: self.ApertureWidth + k * self.ApertureConst] += MaskedScene[:, :, k]
        return temp

    def ApplySourceCube(self, SourceCube):

        MaskedScene = SourceCube * self.Aperture[:, :, np.newaxis]
        return MaskedScene


class MultiShotDispersion:
    def __init__(self, ApertureList, ApertureConst, NumChannel):
        [self.ApertureList, self.ApertureConst, self.NumChannel] = [ApertureList, ApertureConst, NumChannel]
        self.NumShots = len(ApertureList)
        self.ApertureHeight, self.ApertureWidth = self.ApertureList[0].shape
        self.CameraRes = (self.ApertureHeight, self.ApertureWidth + self.ApertureConst*(self.NumChannel - 1))

    def CalcDispersionPattern(self, MaskedSceneList):
        DispersionPatterns = []

        for MaskedScene in MaskedSceneList:
            temp = np.zeros(self.CameraRes)
            for k in range(self.NumChannel):
                temp[:, k * self.ApertureConst: self.ApertureWidth + k * self.ApertureConst] += MaskedScene[:, :, k]
            DispersionPatterns.append(temp)
        return DispersionPatterns

    def ApplySourceCube(self, SourceCube):

        MaskedSceneList = [SourceCube * Aperture[:, :, np.newaxis] for Aperture in self.ApertureList]
        return MaskedSceneList