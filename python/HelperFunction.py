import tkinter.filedialog
import tkinter
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, isdir, join

class Dialog:
    @staticmethod
    def SelectFile():
        fd = os.getcwd()
        root = tkinter.Tk()
        root.withdraw()
        filepath = tkinter.filedialog.askopenfilename(initialdir=f"{fd}/")
        root.destroy()
        return filepath

    @staticmethod
    def Plotting(x, title):

        fig = plt.figure()
        ax = fig.subplots()
        cx = ax.imshow((np.abs(x)), cmap='gist_gray', origin='lower', vmin=0, alpha=1)
        Dialog.forceAspect(ax)
        ax.set_title(title, fontsize=30)
        cbar = fig.colorbar(cx, ax=ax)
        cbar.ax.tick_params(labelsize=15)

    @staticmethod
    def forceAspect(ax, aspect=1):
        im = ax.get_images()
        extent = im[0].get_extent()
        ax.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect)


class FileProcessing:
    @staticmethod
    def readpng(filepath):
        return cv2.imread(filepath)

    @staticmethod
    def showimg(img):
        cv2.imshow('figure', img)
        while True:
            key = cv2.waitKey(0)

            if key & 0xFF == 27:
                break

        cv2.destroyAllWindows()

class DataProcessing:
    @staticmethod
    def CalcPSNR(img_true, img_r, data_range = 1.0):
        img_r = np.clip(img_r, 0, data_range)
        mse = np.mean((img_true - img_r) ** 2)
        if mse == 0:
            return float('inf')
        return 10 * np.log10((data_range ** 2) / mse)
