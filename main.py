from bf_env import BitFlippingEnv
import numpy as np
import channel
import codes
import sarsa
from itertools import product
import matplotlib.pyplot as plt


if __name__ == '__main__':
    ## Define the code ##
    bch_63_45 = codes.BCH('./Hmat/BCH_63_45_std.mat')

    ## Agent ##
    channel = 'BSC'
    noise = 0.2
    env = BitFlippingEnv(bch_63_45, channel, noise)
    Q_sarsa = sarsa.train(env, int(3e6), 0.1)
    # BER = []
    # for snr_i in dB_range:
    #     BER.append(sarsa.test(env, 1000, Q_sarsa, EbN0=snr_i))

    # print(f"BER = {BER}")
    # plt.semilogy(dB_range, BER)
    # plt.show()

    
