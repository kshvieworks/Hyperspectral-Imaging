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