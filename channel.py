import numpy as np

def encode_bits(signal):
    signal = (-1)**signal
    return signal

def decode_bits(signal):
    z = np.sign(signal)
    mapping = {-1: 1, 1: 0}
    z = [mapping[value] for value in z]
    z = np.array(z)
    return z

def AWGN(signal, EbN0, r):
    variance = 1 / (2*r*10**(EbN0/10))
    noise = np.random.normal(0, np.sqrt(variance), signal.shape)
    noisy_signal = encode_bits(signal) + noise
    return decode_bits(noisy_signal)

def BSC(signal, theta):
    rng = np.random.default_rng()
    noise = (rng.random(len(signal)) <= theta).astype(int)
    return (noise + signal) % 2