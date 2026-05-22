import tkinter.filedialog
import numpy as np
import cv2
import os
from os import listdir
from os.path import isfile, isdir, join

class Dialog:
    @staticmethod
    def SelectFile():
        fd = os.getcwd()
        filepath = tkinter.filedialog.askopenfilename(initialdir=f"{fd}/")
        return filepath


class FileProcessing:
    @staticmethod
    def readpng(filepath):
        img = cv2.imread(filepath)
