import gym
from gym import spaces
import numpy as np
import channel
import math
from scipy.stats import norm

class BitFlippingEnv(gym.Env):
    metadata = {'render.modes': ['console']}
    def __init__(self, code, channel_type='BSC', noise=None):
        super(BitFlippingEnv, self).__init__()
        self.code = code
        self.n = code.n
        self.k = code.k
        self.m = code.m
        self.r = code.r
        self.action_space = spaces.Discrete(self.n)
        self.observation_space = spaces.Tuple((spaces.Discrete(2),)*self.m)
        self.nA = self.n
        self.channel_type = channel_type
        self.noise = noise

    def set_noise(self, noise):
        self.noise = noise

    def reset(self):
        if self.channel_type == 'AWGN':
            self.z_double = self.trx_message()
            self.z = channel.decode_bits(self.z_double)
        else:
            self.z = self.trx_message()

        self.num_actions = 0

        self.state = tuple(self.code.syndrome(self.z))
        # if self.channel_type == 'BSC':
        #     llr = np.log((1 - self.noise) / self.noise) * self.z
        # elif self.channel_type == 'AWGN':
            # LLR: 
            #   - +ve value = more likely to be a zero
            #   - -ve value = more likely to be a one
            #   - smaller magnitude = less confident (we should flip)

            # self.sigma2 = 1/(2*self.r*10**(self.noise/10))
            # llr = 2*self.z_double/self.sigma2 # LLRs
            
        # self.llr_avg = np.abs(llr)
        return self.state

    def step(self, action):
        self.num_actions += 1
        reward = 0
        e = np.zeros(self.n, dtype=int)
        e[action] = 1
        self.z = (self.z + e) % 2
        self.state = self.code.syndrome(self.z) # Flip the bit
        
        # Method used in paper (need to review):
        # if self.channel_type == 'BSC':
        #     llr = np.log((1 - self.noise) / self.noise) * self.z
        # elif self.channel_type == 'AWGN':
        #     # if the bit we flipped has a postive LLR then it is a zero
        #     # if the bit we flipped has a negative LLR then it is a one
        #     # BPSK: {0, 1} -> {1, -1}
        #     newbit = 1 if np.sign(self.z_double[action]) > 0 else -1
        #     self.z_double[action] = newbit
        #     llr = 2*self.z/self.sigma2 # LLRs


        # self.llr_avg = (self.num_actions*self.llr_avg + np.abs(llr)) / (self.num_actions + 1)
        # self.path_penalty = - self.llr_avg / (10*np.mean(self.llr_avg))

        reward = -1/10
        done = self.num_actions == 10 or np.all(self.state == 0)
        # reward = self.path_penalty[action]

        if np.all(self.state == 0):
            reward += 1

        return tuple(self.state), reward, done, None

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        print(f"Current state: {self.state}")

    def close(self):
        pass

    def trx_message(self):
        message = np.random.randint(0, 2, size=self.k)
        # message = np.zeros(self.k)
        self.codeword = self.code.encode(message)
        if self.channel_type == 'BSC':
            y = channel.BSC(self.codeword, self.noise)
        elif self.channel_type == 'AWGN':
            y = channel.AWGN(self.codeword, self.noise, self.r)
        return y