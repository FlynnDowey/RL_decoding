import gym
from gym import spaces
import numpy as np
import channel
from scipy.stats import norm

class BitFlippingEnv(gym.Env):
    metadata = {'render.modes': ['console']}
    def __init__(self, code):
        super(BitFlippingEnv, self).__init__()
        # Pass in class that represents a code, for example codes.HammingCode object
        # Pass in the recieved codeword from the channel: z 
        self.code = code
        self.n = code.n
        self.k = code.n

        # Define action and observation space
        # Actions: which bit to flip (including an option not to flip any bit)
        self.action_space = spaces.Discrete(self.n)
        # States: the current syndrome
        self.observation_space = spaces.Tuple((spaces.Discrete(2),)*code.r)

        # Stuff we need
        self.nA = self.n

    def set_EbN0(self, x):
        self.EbN0 = x

    def reset(self):
        # Reset the state of the environment to an initial state
        self.z_true, self.z = trx_message(self.code, self.EbN0)
        self.num_actions = 0
        self.state = tuple(self.code.syndrome(self.z))
        return self.state

    def step(self, action):
        # Execute one time step within the environment
        self.num_actions += 1
        reward = 0
        e = np.zeros(self.n, dtype=int)
        e[action] = 1
        self.z = (self.z + e) % 2
        self.state = self.code.syndrome(self.z) # Flip the bit

        # we need to change this, assuming that we know snr is too optimal?
        reward = -(1/self.num_actions)
        done = np.all(self.state == 0)
        if done:
            reward += 1
        return tuple(self.state), reward, done, None

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        print(f"Current state: {self.state}")

    def close(self):
        pass

def trx_message(code, EbN0):
    # define the message you want to send
    # message = np.random.randint(2, size=k)
    np.random.seed(1)
    # message = np.random.randint(0, 2, size=code.k, dtype=int)
    message = np.zeros(code.k, dtype=int)
    # encode the message i.e. c = Gx
    codeword = code.encode(message)

    # map bits from 0 -> +1, 1 -> -1
    y = channel.encode_bits(codeword)

    # add awgn with EbN0 (dB)
    y = channel.add_noise(y, EbN0, code.k/code.n)

    # map reals to +1/-1
    z = np.sign(y)
    mapping = {-1: 1, 1: 0}
    z = [mapping[value] for value in z]
    z = np.array(z)
    return codeword, z