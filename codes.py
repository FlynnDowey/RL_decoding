import numpy as np
from itertools import product

class HammingCode:
    def __init__(self, n, k, H=None, G=None):
        self.n = n
        self.k = k
        self.r = n - k

        if H is None:
            self.H = self.init_H(self.r)
        else:
            self.H = H
        if G is None:
            self.G = self.init_G(self.r)
        else:
            self.G = G

    def init_H(self,r):
        tmp = list(product([0, 1], repeat=r))
        tmp_arr = np.array(tmp).T
        H = tmp_arr[:, 1:]
        return H
    def init_G(self, r):
        H = np.zero((2**r - 1, 2**r-r-1), dtype=int)
        A = self.H[:, 0:]
        return
    def encode(self, message):
        c = self.G@message.T % 2
        return c
    def syndrome(self, codeword):
        return self.H@codeword.T %2
    