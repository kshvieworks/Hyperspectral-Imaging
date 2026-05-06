import numpy as np
import matplotlib.pyplot as plt


'''''''''''''''''''''''''''''''''''''''''''''Hadamard Mask'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# import CodedAperture as CA
# M = CA.HadamardMask(64, 64)
# M.ApertureFunction()
# S = CA.Shuffled(M.Tx)
# S.Shuffle()
# S.AperturePlot()
# import DispersivePass as DP
# P = DP.Dispersion(S.Tx, 4, 64*4)
# P.ApplySourceSpectrum([1])
# P.Plotting(P.DispersionPattern, 'Coded Aperture Simulation')
# P.Plotting(P.SourceSpreadFunction, 'Incident Spectrum')
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

    def __init__(self, Aperture, ApertureConst, CameraRes):
        [self.Aperture, self.ApertureConst, self.CameraRes] = [Aperture, ApertureConst, CameraRes]

        self.ApertureOrder = self.Aperture.__len__()
        self.SpreadFunction = self.CalcSpreadFunction()
        self.DispersionPattern = self.CalcDispersionPattern()

    def CalcSpreadFunction(self):

        temp = np.ones((self.ApertureOrder, self.CameraRes))
        return np.array([[n if self.ApertureConst*k <= i else 0 for i, n in enumerate(row)] for k, row in enumerate(temp)])

    def CalcDispersionPattern(self):

        return np.dot(self.Aperture, self.SpreadFunction)

    def ApplySourceSpectrum(self, SourceSpectralPos):

        t = int(max(self.ApertureOrder, self.CameraRes))
        self.SourceSpreadFunction = np.zeros((self.ApertureConst*t, self.ApertureConst*t))
        for n in SourceSpectralPos:
            k = self.CameraRes - n

            for c in range(self.ApertureConst):
                self.SourceSpreadFunction[np.arange(k), self.ApertureConst*(np.arange(k) + n) + c] += 1

        self.SourceSpreadFunction = self.SourceSpreadFunction[:self.ApertureOrder, :self.CameraRes]

        self.DispersionPattern = np.dot(self.Aperture, self.SourceSpreadFunction)

        self.IncidentSpectrum = np.zeros((self.ApertureOrder, self.CameraRes))
        for n in SourceSpectralPos:
            for c in range(self.ApertureConst):
                self.IncidentSpectrum[:, self.ApertureConst*n + c] += 1

    def Plotting(self, x, title):

        fig = plt.figure()
        ax = fig.subplots()
        cx = ax.imshow((np.abs(x)), cmap='gist_gray', origin='lower', vmin=0, vmax=1, alpha=1)
        self.forceAspect(ax)
        ax.set_title(title, fontsize=30)
        cbar = fig.colorbar(cx, ax=ax)
        cbar.ax.tick_params(labelsize=15)


    def forceAspect(self, ax, aspect=1):
        im = ax.get_images()
        extent = im[0].get_extent()
        ax.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect)