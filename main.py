from bf_env import BitFlippingEnv
import numpy as np
import channel
import codes
import sarsa
import sarsa_param
import matplotlib.pyplot as plt
import os
from scipy.io import loadmat

import os
from pathlib import Path

from utils import (
    main,
    handle,
    run,
    plot_BER,
)
os.chdir(Path(__file__).parent.resolve())

# BCH agent (loads G and H)
def BCH_agent():
    bch_63_45 = codes.BCH('./Hmat/BCH_63_45_std.mat')
    return bch_63_45

# Reed-Muller agent (loads G and H)
def RM_agent(label):
    rm = codes.RM('./Hmat/RM_' + label[0] + '_' + label[1] + '_' + 'std.mat')
    return rm

def fit(agent, channel, noise, type=None):
    """
    fit trains an RL agent using sarsa.

    parameters:
    agent:      RM_agent or BCH_agent
    channel:    BSC or AWGN
    noise:      level of noise used in training. Units need to match with channel selection.
    type:       tabular or param. tabular is traditional sarsa using a q-table. param approximates
                the q function using a simple neural network.

    returns:    optimal q table if type == tabular, o/w weights of NN if type == param (these parameters are also stored
                as class variables of env).
    """
    env = BitFlippingEnv(agent, channel, noise)
    if type == 'tabular':
        Q = sarsa.train(env, int(3e6), 0.1)
        return Q, env
    elif type == 'param':
        w, v = sarsa_param.train(env, int(3e5), 0.01)
        return w, v, env
    else:
        raise ValueError("Invalid type. Specify type as tabular or param")

def eval(env, channel, type):
    """
    eval tests an RL agent by computing bit error rates (BER).

    parameters:
    env:        environment defined in bf_env.py; needs to hold weights or q-table
    channel:    BSC or AWGN
    type:       tabular or param; agent holds its policy in a neural network or q-table

    returns:
    BER:        numpy array of BER for 10 different snr values.
    """

    dB_range = None
    if channel == 'BSC':
        dB_range = np.linspace(0.01, 0.45, 10)
    elif channel == 'AWGN':
        dB_range = np.linspace(1, 7, 10) 

    function_map = {
        'tabular': lambda snr: sarsa.test(env, 1000, env.Q, EbN0=snr),
        'param': lambda snr: sarsa_param.test(env, 1000, env.w, env.v, EbN0=snr)
    }

    BER = []
    for snr_i in dB_range:
        BER.append(function_map[type](snr_i))
    return BER


############################################## Driver code ##########################################################
# 1. BCH with BSC
@handle("bch-bsc")
def BHC_BSC():
    # Define the code 
    code_type = 'BCH'
    code_label = ('63', '45')
    channel = 'BSC'
    noise = 0.2 # training noise
    dB_range = np.linspace(0.01, 0.45, 10) # testing noise

    # saving
    figure_name = "BER_sarsa" + code_type + "_" + code_label[0] + "_" + code_label[1] + "_" + channel

    # tabular setting
    agent_tabular = BCH_agent()
    _, env_tab = fit(agent_tabular, channel, noise, type='tabular')
    BER_tabular = eval(env_tab, channel, type='tabular')

    # parameterized setting
    agent_param = BCH_agent()
    _, _, env_param = fit(agent_param, channel, noise, type='param')
    BER_param = eval(env_param, channel, type='param')

    # load matlab decoder BER, store in 'bench'
    mat = loadmat('./benchmark/BER_BCH_BSC_63_45.mat')
    bench = mat['BER']
    bench = bench[:, 0]

    # labeling figures
    name = channel + " " + code_type + "[" + code_label[0] + ", " + code_label[1] + "]" 
    x_range = [1-x for x in dB_range]

    # plot results
    plot_BER(x_range, BER_tabular, BER_param, bench, '1 - prob. error in BSC', name, figure_name)

####################################################################################################################
# 2. RM with BSC
@handle("rm-bsc")
def RM_BSC():
    # Define the code 
    code_type = 'RM'
    code_label = ('3', '6')
    channel = 'BSC'
    noise = 0.2 # training noise
    dB_range = np.linspace(0.01, 0.45, 10) # testing noise

    # saving
    figure_name = "BER_sarsa" + code_type + "_" + code_label[0] + "_" + code_label[1] + "_" + channel

    # tabular setting
    agent_tabular = BCH_agent()
    _, env_tab = fit(agent_tabular, channel, noise, type='tabular')
    BER_tabular = eval(env_tab, channel, type='tabular')

    # parameterized setting
    agent_param = BCH_agent()
    _, _, env_param = fit(agent_param, channel, noise, type='param')
    BER_param = eval(env_param, channel, type='param')

    # load matlab decoder BER, store in 'bench'
    mat = loadmat('./benchmark/BER_RM_BSC_3_6.mat')
    bench = mat['BER']
    bench = bench[:, 0]

    # labeling figures
    name = channel + " " + code_type + "[" + code_label[0] + ", " + code_label[1] + "]" 
    x_range = [1-x for x in dB_range]

    # plot results
    plot_BER(x_range, BER_tabular, BER_param, bench, '1 - prob. error in BSC', name, figure_name)

####################################################################################################################
# 4. BCH with AWGN
@handle("bch-awgn")
def BHC_AWGN():
    # Define the code 
    code_type = 'BCH'
    code_label = ('63', '45')
    channel = 'AWGN'
    noise = 2 # training noise
    dB_range = np.linspace(1, 7, 10) # testing noise

    # saving
    figure_name = "BER_sarsa" + code_type + "_" + code_label[0] + "_" + code_label[1] + "_" + channel

    # tabular setting
    agent_tabular = BCH_agent()
    _, env_tab = fit(agent_tabular, channel, noise, type='tabular')
    BER_tabular = eval(env_tab, channel, type='tabular')

    # parameterized setting
    agent_param = BCH_agent()
    _, _, env_param = fit(agent_param, channel, noise, type='param')
    BER_param = eval(env_param, channel, type='param')

    # load matlab decoder BER, store in 'bench'
    mat = loadmat('./benchmark/BER_BCH_AWGN_63_45.mat')
    bench = mat['BER']
    bench = bench[:, 0]

    # labeling figures
    name = channel + " " + code_type + "[" + code_label[0] + ", " + code_label[1] + "]" 

    # plot results
    plot_BER(dB_range, BER_tabular, BER_param, bench, 'SNR (dB)', name, figure_name)

####################################################################################################################
# 5. RM with AWGN
@handle("rm-awgn")
def RM_AWGN():
    # Define the code 
    code_type = 'RM'
    code_label = ('3', '6')
    channel = 'AWGN'
    noise = 2 # training noise
    dB_range = np.linspace(1, 7, 10) # testing noise

    # saving
    figure_name = "BER_sarsa" + code_type + "_" + code_label[0] + "_" + code_label[1] + "_" + channel

    # tabular setting
    agent_tabular = BCH_agent()
    _, env_tab = fit(agent_tabular, channel, noise, type='tabular')
    BER_tabular = eval(env_tab, channel, type='tabular')

    # parameterized setting
    agent_param = BCH_agent()
    _, _, env_param = fit(agent_param, channel, noise, type='param')
    BER_param = eval(env_param, channel, type='param')

    # load matlab decoder BER, store in 'bench'
    mat = loadmat('./benchmark/BER_RM_AWGN_3_6.mat')
    bench = mat['BER']
    bench = bench[:, 0]

    # labeling figures
    name = channel + " " + code_type + "[" + code_label[0] + ", " + code_label[1] + "]" 

    # plot results
    plot_BER(dB_range, BER_tabular, BER_param, bench, 'SNR (dB)', name, figure_name)

if __name__ == "__main__":
    main()