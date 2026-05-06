import numpy as np
from scipy.optimize import nnls
from scipy.ndimage import shift
import matplotlib.pyplot as plt


'''''''''''''''''''''''''''''''''''''''''''''Hadamard Mask'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# import CodedAperture as CA
# M = CA.HadamardMask(64, 64)
# M.ApertureFunction()
# S = CA.Shuffled(M.Tx)
# S.Shuffle()
# S.AperturePlot()
# import DispersivePass as DP
# P = DP.Dispersion(S.Tx, 4, 1000)
# P.ApplySourceSpectrum([10, 100, 200])
# P.Plotting(P.DispersionPattern, 'Coded Aperture Simulation')
# P.Plotting(P.IncidentSpectrum, 'Incident Spectrum')
# import ReconstructionSpectrum as RS
# X = RS.NonNegativeLeastSquare(P.ApertureConst, P.CameraRes)
# X.CalcNNLS(S.Tx, P.DispersionPattern)
# X.Plotting(X.x, 'Reconstructed Spectrum')
# X.AlignSpectrum()
# X.Plotting(X.x, 'Aligned Reconstructed Spectrum')
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


class NonNegativeLeastSquare:

    def __init__(self, ApertureConst, CameraRes):

        self.ApertureConst, self.CameraRes = ApertureConst, CameraRes

    def CalcNNLS(self, A, b):

        self.x = np.array([nnls(A, b[:,k])[0] for k in range(self.CameraRes)]).T

    def AlignSpectrum(self):

        for k in range(self.x.shape[0]):

            self.x[k, :] = shift(self.x[k, :], -self.ApertureConst*(k+1), cval=0)


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