from bf_env import BitFlippingEnv
import numpy as np
import channel
import codes
import sarsa
from itertools import product


if __name__ == '__main__':
    ## Define the code ##
    dB_range=[1,2,3,4,5,6,7,8]
    n, k = (63, 45)
    bch_63_45 = codes.BCH(n, k)

    ## Agent ##
    env = BitFlippingEnv(bch_63_45)
    Q_sarsa = sarsa.train(env, 500, 0.1, 1, 0.95)
    BER = sarsa.test(env, 10, Q_sarsa)

    print(f"BER = {BER}")

    # tmp = list(product([0, 1], repeat=bch_63_45.r))
    # tmp_arr = np.array(tmp)
    # states = []
    # for state in tmp_arr[1:, :]:
    #     states.append(tuple(state))

    # policy = []
    # for key in states:
    #     policy.append(np.argmax(Q_sarsa[key]))
    
