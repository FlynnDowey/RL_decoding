import numpy as np
from collections import defaultdict, deque
import sys
import matplotlib.pyplot as plt

def epsilon_greedy(Q, state, nA, eps):
    if np.random.random() > eps: # select greedy action with probability epsilon
        return np.argmax([Q(state, a) for a in range(nA)])
    else:                     # otherwise, select an action randomly
        return np.random.choice(np.arange(nA))
    
def feature_vector(state, a, n_features):
    x = np.zeros(n_features) # feature vector
    syndrome, residual = state

    x[:len(syndrome)] = syndrome  # fill first bits with syndrome
    x[len(syndrome)] = residual # this is the 'residual' term
    x[len(syndrome) + a] = 1
    return x


def train(env, num_episodes, alpha, mov_avg=1000, gamma=0.99, EbN0=1):
    w = np.zeros(env.n + 1 + env.m) 
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
        action = epsilon_greedy(lambda s, a: w@feature_vector(s, a, env.m + 1 + nA), state, nA, eps)
        total_reward = 0
        while True:
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            if not done:
                next_action = epsilon_greedy(lambda s, a: w@feature_vector(s, a, env.m + 1 + nA), next_state, nA, eps)
                x = feature_vector(state, action, env.m + 1 + nA)
                x_prime = feature_vector(next_state, next_action, env.m + 1 + nA)
                td_error =reward + gamma*w@x_prime - w@x
                w += alpha*td_error*x
                state = next_state
                action = next_action
            if done:
                td_error =reward + gamma*w@x
                w += alpha*td_error*x
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
        function = lambda s, a: w@feature_vector(s, a, env.m + 1 + env.n)
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