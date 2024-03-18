import gym
from gym import spaces
import numpy as np
from scipy.stats import norm

class BitFlippingEnv(gym.Env):
    metadata = {'render.modes': ['console']}
    def __init__(self, code, z):
        super(BitFlippingEnv, self).__init__()
        # Pass in class that represents a code, for example codes.HammingCode object
        # Pass in the recieved codeword from the channel: z 
        self.code = code
        self.n = code.n
        self.k = code.n

        # Define action and observation space
        # Actions: which bit to flip (including an option not to flip any bit)
        self.action_space = spaces.Discrete(self.n + 1)
        # States: the current syndrome
        self.observation_space = spaces.MultiBinary(code.r)

        # Initial state
        self.state = code.syndrome(z)
        self.z_init = z
        self.z = z

        # Stuff we need
        self.nA = self.n + 1

    def reset(self):
        # Reset the state of the environment to an initial state
        self.state = self.code.syndrome(self.z_init)
        return self.state

    def step(self, action):
        # Execute one time step within the environment
        reward = 0
        if action < self.n:
            e = np.zeros(self.n, dtype=int)
            e[action] = 1
            self.z = (self.z + e) % 2
            self.state = self.code.syndrome(self.z) # Flip the bit

            # we need to change this, assuming that we know snr is too optimal?
        reward = -np.sum(self.state)
        done = np.all(self.state == 0)
        if done:
            reward += 1
        return self.state, reward, done, None

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        print(f"Current state: {self.state}")

    def close(self):
        pass
