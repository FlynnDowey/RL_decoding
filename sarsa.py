import numpy as np
from collections import defaultdict, deque
import sys
import matplotlib.pyplot as plt

def binary_to_decimal(binary_vector):
    """Convert a binary vector to a decimal number."""
    return int(np.dot(binary_vector, 1 << np.arange(binary_vector.size)[::-1]))

def epsilon_greedy(Q, state, nA, eps):
    if np.random.random() > eps: # select greedy action with probability epsilon
        return np.argmax(Q[state])
    else:                     # otherwise, select an action randomly
        return np.random.choice(np.arange(nA))

def sarsa(env, num_episodes, alpha, mov_avg, gamma=1.0):
    # initialize action-value function (empty dictionary of arrays)
    Q = defaultdict(lambda: np.zeros(env.nA))
    nA = env.nA
    tmp_scores = deque(maxlen=10)
    avg_scores = deque(maxlen=num_episodes)

    # initialize performance monitor
    # loop over episodes
    eps = 0.9
    for i_episode in range(1, num_episodes+1):
        eps = eps*np.exp(-i_episode)
        # monitor progress
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()   
        
        ## TODO: complete the function
        state = tuple(env.reset())

        action = epsilon_greedy(Q, state, nA, eps)
        total_reward = 0
        while True:
            next_state, reward, done, _ = env.step(action)
            next_state = tuple(next_state)
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