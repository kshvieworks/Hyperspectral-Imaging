import numpy as np
import matplotlib.pyplot as plt

class RectAperture:

    def __init__(self, a, b, nx, ny, R, wl):

        [self.a, self.b, self.nx, self.ny, self.R, self.wl] = [a, b, nx, ny, R, wl]

        self.sx = 1000 * self.a * self.nx
        self.sy = 1000 * self.a * self.ny

        self.S = np.zeros((self.sx, self.sy))