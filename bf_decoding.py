import numpy as np
n = 7; k = 4 # hamming code (7, 4)
rate = k / n

noise = 1
sigma2 = 1/(2*rate*10**(noise/10))

def encode_bits(signal):
    signal = (-1)**signal
    return signal

H = np.array([[1, 1, 0, 1, 1, 0, 0],[1, 0, 1, 1, 0, 1, 0], [0, 1, 1, 1, 0, 0, 1]], dtype=int)
G = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 1]], dtype=int)
x = np.array([0, 1, 1, 0], dtype=int)
c = G@x.T % 2

z_BPSK = np.random.randn(7)
c_BPSK = encode_bits(c)

llr = 2*z_BPSK/sigma2

print(f"true codeword c = ({c_BPSK})")
print(f"recieved codeword z = ({z_BPSK})")
print(f"{llr}")

z_BPSK[4] = -1
llr = 2*z_BPSK/sigma2
print(f"{llr}")

