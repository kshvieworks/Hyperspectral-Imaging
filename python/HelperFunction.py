import tkinter.filedialog
import tkinter
import numpy as np
import cv2
import os
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

