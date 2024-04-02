## About
This repository implements the results found in [Reinforcement Learning for Channel Coding: Learned Bit-Flipping Decoding](https://arxiv.org/abs/1906.04448)
## Preliminaries 
Create a venv:
1. ``` pip install virtualenv ```
2. ``` python -m venv env ```
3. ``` source env/bin/activate ```

Download requirements.txt:
1. ```pip install -r /path/to/requirements.txt```

 ## Files 
- ```main.py ``` driver code to train/evaluate the agent and compare results with benchmark
- ```bf_env.py``` is an OpenAI gym environment for bit flipping environment
- ```sarsa.py``` runs sarsa algorithm using e-greedy policy 
- ```bf_decoding.py``` (old)
- ```channel.py``` models AWGN and BSC channels 
- ```codes.py``` is used to load generator and parity check matrices to define the type of code the agent will be decoding

## Folders
- ```/Hmat``` holds generator and parity check matricies for codes in .mat format; see [here](https://github.com/fabriziocarpi/RLdecoding)
- ```/MATLAB``` runs benchmark decoding algorithms in matlab
- ```/policies``` saves the Q table for a specific code after training the agent using sarsa (too large for repository)
- ```/benchmark``` holds .mat files for BER using MATLAB decoding algorithms
- ```/figs``` figures from training and BER comparison
