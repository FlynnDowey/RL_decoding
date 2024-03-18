import numpy as np
from scipy.io import loadmat
from itertools import product

class BlockCode:
    def __init__(self, n, k):
        self.n = n
        self.k = k
        self.r = n-k
        self.H, self.G = self.init_H_and_G(n, k)
        self.filename = None

    def init_H_and_G(self, n, k):
        mat = loadmat(self.filename)
        H = mat['H']
        G = mat['G'].T
        return H, G
    
    def encode(self, message):
        c = self.G@message.T % 2
        return c
    def syndrome(self, codeword):
        return self.H@codeword.T %2
    
class BCH(BlockCode):
    def __init__(self, n, k):
        self.filename = 'Hmat/BCH_'+ str(n) + '_' + str(k) + '_std.mat'
        super().__init__(n, k)


class RM(BlockCode):
    def __init__(self, n, k):
        super().__init__(n, k)
        self.filename = 'Hmat/RM_'+ str(n) + '_' + str(k) + '_std.mat'
    
## Old Code (not used) ##

class HammingCode(BlockCode):
    def __init__(self, n, k):
        super().__init__(n, k)
        
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