import numpy as np

def encode_bits(signal):
    signal = (-1)**signal
    return signal

def add_noise(signal, EbN0, r):
    # signal: bit stream of length n
    # snr: rate * snr := k*Eb/(N*N0)
    variance = 1 / (2*r*10**(EbN0/10))
    noise = np.random.normal(0, np.sqrt(variance), signal.shape)
    noisy_signal = signal + noise
    return noisy_signal

def bernoulli(signal):
    theta=0.2
    rng = np.random.default_rng()
    noise = (rng.random(len(signal)) <= theta).astype(int)
    return (noise + signal) % 2