import gym
from gym import spaces
import numpy as np
from scipy.stats import norm

class BitFlippingEnv(gym.Env):
    metadata = {'render.modes': ['console']}
    def __init__(self, H, z, n=7, k=3,):
        super(BitFlippingEnv, self).__init__()

        # Define action and observation space
        # Actions: which bit to flip (including an option not to flip any bit)
        self.action_space = spaces.Discrete(n + 1)
        # States: the current syndrome
        self.observation_space = spaces.MultiBinary(n-k)

        # Initial state
        self.state = H@z.T % 2
        self.H = H
        self.z_init = z
        self.z = z
        self.n = n
        self.k = k

        # Stuff we need
        self.nA = n + 1

    def reset(self):
        # Reset the state of the environment to an initial state
        self.state = self.H@self.z_init.T % 2
        return self.state

    def step(self, action):
        # Execute one time step within the environment
        if action < self.n:
            e = np.zeros(self.n, dtype=int)
            e[action] = 1
            self.z = (self.z + e) % 2
            self.state = (self.H@(self.z).T) % 2 # Flip the bit

        r = self.k/self.n
        std = np.sqrt(1/(2*r*20)); #SNR = 20
        reward = -np.abs(np.log(norm.pdf(self.z, 0, std) / norm.pdf(self.z, 1, std)))
        # done = np.all(self.state) == 0 
        done = self.state[0] == 0 and self.state[1] == 0 and self.state[2] == 0

        return self.state, reward, done, None

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        print(f"Current state: {self.state}")

    def close(self):
        pass
