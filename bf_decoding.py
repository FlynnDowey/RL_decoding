import numpy as np
n = 7; k = 4 # hamming code (7, 4)
rate = k / n
theta=0.2

rng = np.random.default_rng()
noise = (rng.random(n) <= theta).astype(int)

H = np.array([[1, 1, 0, 1, 1, 0, 0],[1, 0, 1, 1, 0, 1, 0], [0, 1, 1, 1, 0, 0, 1]], dtype=int)
G = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 1]], dtype=int)
x = np.array([0, 1, 1, 0], dtype=int)
c = G@x.T % 2

w = np.array([0, 0, 1, 0, 0, 0, 0], dtype=int)
z = (c + w) % 2

print(f"true codeword c = ({c})")
print(f"recieved codeword z = ({z})")

Q = np.zeros(n, dtype=int)
while np.any(H@z.T % 2)!= 0:
    s = H@z.T % 2 # syndrome
    V = np.sum(s) # sum the number of 1s in Hc
    for i in range(n):
        e = np.zeros(n, dtype=int)
        e[i] = 1
        _s = H@(z + e).T % 2
        Q[i] = V - np.sum(_s) # store the most likely index that an error occured

    e = np.zeros(n, dtype=int)
    # correct the error
    e[np.argmax(Q)] = 1 
    z = (z + e)%2
