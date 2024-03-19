import numpy as np
from collections import defaultdict, deque
import sys
import matplotlib.pyplot as plt

def epsilon_greedy(Q, state, nA, eps):
    if np.random.random() > eps: # select greedy action with probability epsilon
        return np.argmax(Q[state])
    else:                     # otherwise, select an action randomly
        return np.random.choice(np.arange(nA))

def train(env, num_episodes, alpha, mov_avg, gamma=1.0, EbN0=1):
    # initialize action-value function (empty dictionary of arrays)
    Q = defaultdict(lambda: np.zeros(env.nA))
    nA = env.nA
    tmp_scores = deque(maxlen=10)
    avg_scores = deque(maxlen=num_episodes)

    # initialize performance monitor
    # loop over episodes
    eps = 0.9
    for i_episode in range(1, num_episodes+1):
        if i_episode % 10 == 0:
            eps = max(eps*np.exp(-0.01*i_episode), 0.01)

        # monitor progress
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()   
        
        ## TODO: complete the function
        env.set_EbN0(EbN0)
        state = env.reset()

        action = epsilon_greedy(Q, state, nA, eps)
        total_reward = 0
        while True:
            next_state, reward, done, _ = env.step(action)
            next_state = next_state
            total_reward += reward
            if not done:
                next_action = epsilon_greedy(Q, next_state, nA, eps)
                td_error = alpha*(reward + gamma*Q[next_state][next_action] - Q[state][action])
                Q[state][action] += td_error
                state = next_state
                action = next_action
            if done:
                td_error = alpha*(reward - Q[state][action])
                Q[state][action] += td_error
                tmp_scores.append(total_reward)
                break
        if (i_episode % mov_avg == 0):
            avg_scores.append(np.mean(tmp_scores))
    # plot performance
    plt.plot(np.linspace(0,num_episodes,len(avg_scores), endpoint=False), np.asarray(avg_scores))
    plt.xlabel('Episode Number')
    plt.ylabel('Average Reward (Over Next %d Episodes)' % mov_avg)
    plt.show()
    return Q

def test(env, num_runs, optimal_Q, EbN0=1):
    BER = 0
    policy = lambda state : np.argmax(optimal_Q[state])
    env.set_EbN0(EbN0)
    for iter in range(num_runs):
        state = env.reset()
        if np.all(optimal_Q[state]) == 0:
            print("Agent has not seen codeword yet.")
            break
        action = policy(state)
        while True:
            next_state, _, done, _ = env.step(action)
            next_state = next_state
            if not done:
                next_action = policy(next_state)
                state = next_state
                action = next_action
            if done:
                break
        BER += np.sum(env.z ^ env.z_true) / len(env.z)
    return BER/num_runs