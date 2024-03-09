from bf_env import BitFlippingEnv
import numpy as np

if __name__ == '__main__':
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
    env = BitFlippingEnv(H, z, n, k)
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()  # Randomly choose an action
        obs, reward, done, info = env.step(action)
        env.render()
