import numpy as np
from scipy.io import loadmat

class BlockCode:
    def __init__(self, filename=None):
        self.filename = filename
        self.load_code()

    def load_code(self):
        mat = loadmat(self.filename)
        self.H = np.int64(mat['H'])
        self.G = np.int64(mat['G'].T)
        self.n = self.H.shape[1]
        self.m = self.H.shape[0]
        self.k = self.n - self.m
        self.r = self.k / self.n

    def encode(self, message):
        c = self.G@message.T % 2
        return c
    def syndrome(self, codeword):
        return self.H@codeword.T %2
    
class BCH(BlockCode):
    def __init__(self, filename):
        super().__init__(filename)

class RM(BlockCode):
    def __init__(self, filename):
        super().__init__(filename)
    
class HammingCode(BlockCode):
    def __init__(self, filename):
        super().__init__(filename)