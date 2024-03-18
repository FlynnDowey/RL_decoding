from bf_env import BitFlippingEnv
import numpy as np
import channel
import codes
from sarsa import sarsa
from itertools import product

def hamming_code():
    n = 7; k = 4 # hamming code (7, 4)
    rate = k / n

    # parity check and generator matrix for (7,4) hamming code
    H = np.array([[1, 1, 0, 1, 1, 0, 0],[1, 0, 1, 1, 0, 1, 0], [0, 1, 1, 1, 0, 0, 1]], dtype=int)
    G = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 1]], dtype=int)

    # create hamming code object using n = 7, k = 4, H and G
    hc = codes.HammingCode(n, k, H, G)


if __name__ == '__main__':
    ## Define the code ##
    n, k = (63, 45)
    bch_63_45 = codes.BCH(n, k)

    # define the message you want to send
    # message = np.random.randint(2, size=k)
    message = np.zeros(k, dtype=int)
    # encode the message i.e. c = Gx
    codeword = bch_63_45.encode(message)

    # map bits from 0 -> +1, 1 -> -1
    y = channel.encode_bits(codeword)

    # add awgn with EbN0 (dB)
    EbN0 = -20
    y = channel.add_noise(y, EbN0, bch_63_45.r)

    # map reals to +1/-1
    z = np.sign(y)
    mapping = {-1: 1, 1: 0}
    z = [mapping[value] for value in z]
    z = np.array(z)

    # if no error (or recieved codeword)
    if np.all(bch_63_45.syndrome(z) == 0):
        print(f"done: {bch_63_45.syndrome(z)}")
    else:

        ## Agent ##
        env = BitFlippingEnv(bch_63_45, z)
        Q_sarsa = sarsa(env, 500, 0.1, 10)

        tmp = list(product([0, 1], repeat=bch_63_45.r))
        tmp_arr = np.array(tmp)
        states = []
        for state in tmp_arr[1:, :]:
            states.append(tuple(state))

        policy = []
        for key in states:
            policy.append(np.argmax(Q_sarsa[key]))

        c_bar = np.copy(z)
        for action in policy:
            e = np.zeros_like(z)
            if action > len(z):
                continue
            else:
                e[action] = 1
                c_bar = (c_bar + e) % 2

        print(f"number of errors = {np.sum(c_bar ^ codeword)}")
