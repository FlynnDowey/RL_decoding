from bf_env import BitFlippingEnv
import numpy as np
import channel
import codes
import sarsa
import matplotlib.pyplot as plt
import dill as pickle
import os
from scipy.io import loadmat


def BCH_agent():
    bch_63_45 = codes.BCH('./Hmat/BCH_63_45_std.mat')
    return bch_63_45

def RM_agent(label):
    rm = codes.RM('./Hmat/RM_' + label[0] + '_' + label[1] + '_' + 'std.mat')
    return rm

def plot_BER(x_vars, y_vars, units, title):
    mat = loadmat('./benchmark/BER_BCH_63_45.mat')
    bench = mat['BER']
    bench = bench[:, 0]

    plt.semilogy(x_vars, y_vars, marker='o', fillstyle='none', linestyle='--', linewidth=1.5, markersize=8, label='Agent')
    plt.semilogy(x_vars, bench, marker='o', fillstyle='none', linestyle='--', linewidth=1.5, markersize=8, label='Matlab')
    plt.xlabel('Noise (' + units + ')')
    plt.ylabel('BER')
    plt.title(title)
    plt.grid(visible=True, which='both', axis='y')
    plt.grid(visible=True, which='major', axis='x')
    plt.savefig('./figs/' + figure_name + '.png')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    train = False

    ## Define the code ##
    code_type = 'BCH'
    code_label = ('63', '45')

    ## Define channel characteristics ##
    channel = 'BSC'
    noise = 0.2
    # dB_range = np.linspace(1, 4, 10) # (dB) or probabilities
    dB_range = np.linspace(0.01, 0.45, 10)

    ## saving ##
    policy_name = "Q_sarsa_" + code_type + "_" + code_label[0] + "_" + code_label[1] + "_" + channel
    figure_name = "BER_sarsa_" + code_type + "_" + code_label[0] + "_" + code_label[1] + "_" + channel
    ## Get code environment ##
    if code_type == 'RM':
        agent = RM_agent(code_label)
    elif code_type == 'BCH':
        agent = BCH_agent()
    
    env = BitFlippingEnv(agent, channel, noise)

    if os.path.exists('./policies/' + policy_name + '.pkl') and train == False:
        with open('./policies/' + policy_name + '.pkl', 'rb') as file:
            Q_sarsa = pickle.load(file)
    elif train == True:
        Q_sarsa = sarsa.train(env, int(3e6), 0.1)
        with open('./policies/' + policy_name + '.pkl', 'wb') as file:
            pickle.dump(Q_sarsa, file)
    
    ## Evaluate model ##
    BER = []
    for snr_i in dB_range:
        BER.append(sarsa.test(env, 1000, Q_sarsa, EbN0=snr_i))

    ## Plotting ##
    name = channel + " " + code_type + "[" + code_label[0] + ", " + code_label[1] + "]"
    if channel == 'BSC':
        x_range = [1-x for x in dB_range]
        plot_BER(x_range, BER, '1 - prob. error in BSC', name)
    elif channel == 'AWGN':
        plot_BER(dB_range, BER, '(dB)', name)

    
