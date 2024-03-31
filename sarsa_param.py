import numpy as np
from collections import defaultdict, deque
import sys
import matplotlib.pyplot as plt

def epsilon_greedy(Q, state, nA, eps):
    if np.random.random() > eps: # select greedy action with probability epsilon
        return np.argmax([Q(state, a) for a in range(nA)])
    else:                     # otherwise, select an action randomly
        return np.random.choice(np.arange(nA))
    
def feature_vector_simple(state, a, n_features):
    x = np.zeros(n_features) # feature vector
    syndrome, residual = state

    x[:len(syndrome)] = syndrome  # fill first bits with syndrome
    x[len(syndrome)] = residual # this is the 'residual' term
    x[len(syndrome) + a] = 1
    return x

def feature_vector_nn(state, a, n_features):
    x = np.zeros(n_features) # feature vector
    syndrome, residual = state
    x[:len(syndrome)] = syndrome  # fill first bits with syndrome
    x[len(syndrome)] = residual # this is the 'residual' term
    x[len(syndrome) + 1] = a
    x[-1] = 1 # bias
    return x

def train(env, num_episodes, alpha, mov_avg=1000, gamma=0.99, EbN0=1):
    w = np.zeros((32,3 + env.m)) 
    v = np.zeros((32, env.n))
    nA = env.nA
    tmp_scores = deque(maxlen=mov_avg)
    avg_scores = deque(maxlen=num_episodes)
    eps = 0.9

    for i_episode in range(1, num_episodes+1):
        if i_episode % 1000 == 0:
            eps = max(eps*0.9, 1e-3)
            alpha = max(alpha*0.9, 1e-6)

        # monitor progress
        if i_episode % 10000 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()   
        
        state = env.reset()
        action = epsilon_greedy(lambda s, a: w@feature_vector_nn(s, a, env.m + 3), state, nA, eps)


        total_reward = 0
        while True:
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            if not done:
                next_action = epsilon_greedy(lambda s, a: w@feature_vector_nn(s, a, env.m + 3), next_state, nA, eps)
                x = feature_vector_nn(state, action, env.m + 3)
                x_prime = feature_vector_nn(next_state, next_action, env.m + 3)

                _yprime = w@x_prime
                _y = w@x

                _activations_prime = np.array([np.tanh(_yprime[i]) for i in range(_yprime.shape[0])])
                _activations = np.array([np.tanh(_y[i]) for i in range(_y.shape[0])])

                td_error =reward + gamma*v.T@_activations_prime - v.T@_activations
                grad_w = gamma*v.T@(1 - _activations_prime**2) - (1 - _activations**2)
                grad_v = gamma*_activations_prime - _activations

                w += alpha*td_error*grad_w
                v += alpha*td_error*grad_v
                state = next_state
                action = next_action
            if done:
                td_error =reward + gamma*np.tanh(w@x)
                grad_w = gamma*(1 - w@x**2)
                grad_v = gamma*np.tanh(w@x)
                w += alpha*td_error*grad_w
                v += alpha*td_error*grad_v
                tmp_scores.append(total_reward)
                break

        if (i_episode % mov_avg == 0):
            avg_scores.append(np.mean(tmp_scores))
    
    # plot performance      
    plt.plot(np.linspace(0,num_episodes,len(avg_scores), endpoint=False), np.asarray(avg_scores))
    plt.xlabel('Episode Number')
    plt.ylabel('Average Reward (Over Next %d Episodes)' % mov_avg)
    plt.savefig('./figs/reward_fun_sarsa_RM_3_6.png')
    plt.show()
    print(('Best Average Reward over %d Episodes: ' % mov_avg), np.max(avg_scores))    
    return w

def test(env, num_runs, w, EbN0=0.1):
    BER = 0
    env.set_noise(EbN0)
    max_iters = 10
    for iter in range(num_runs):
        state = env.reset()
        function = lambda s, a: w@feature_vector_nn(s, a, env.m + 3)
        action = np.argmax([function(state, a) for a in range(env.n)])
        i = 0
        while True:
            next_state, _, done, _ = env.step(action)
            if not done and i < max_iters:
                next_action = np.argmax([function(next_state, a) for a in range(env.n)])
                state = next_state
                action = next_action
            if done:
                break
            if i >= max_iters:
                print("Agent unable to decode")
                break
            i += 1
        BER += np.sum(env.z.astype(int) ^ env.codeword.astype(int)) / len(env.z)
    return BER/num_runs