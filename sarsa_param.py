import numpy as np
from collections import defaultdict, deque
import sys
import matplotlib.pyplot as plt
import torch

def epsilon_greedy(Q, state, nA, eps):
    if np.random.random() > eps: # select greedy action with probability epsilon
        return np.argmax([Q(state, a) for a in range(nA)])
    else:                     # otherwise, select an action randomly
        return np.random.choice(np.arange(nA))
    
def feature_vector_nn(state, a, n_features):
    x = np.zeros(n_features) # feature vector
    syndrome, residual = state
    x[:len(syndrome)] = syndrome  # fill first bits with syndrome
    x[len(syndrome)] = residual # this is the 'residual' term
    x[len(syndrome) + 1] = a
    x[-1] = 1 # bias
    return x

def train(env, num_episodes, alpha, mov_avg=1000, gamma=0.99, EbN0=1):
    ## Dimension of NN ##
    # y = V*h(Wx) where
    # - V is nxk
    # - W is kxd
    # - d is dx1
    n = env.n
    k = 32
    d = env.m + 3

    # weights
    w = torch.randn(k, d, requires_grad=True)
    v = torch.randn(n, k, requires_grad=True)

    # solution to exploding/vanishing gradients
    torch.nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
    torch.nn.init.kaiming_uniform_(v, mode='fan_in', nonlinearity='relu')

    nA = env.nA
    tmp_scores = deque(maxlen=mov_avg)
    avg_scores = deque(maxlen=num_episodes)
    eps = 0.9

    optimizer = torch.optim.SGD([w, v], lr=alpha)  # Use an optimizer to handle weight updates

    for i_episode in range(1, num_episodes+1):
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()   
            eps = max(eps*0.9, 1e-3)
            for param_group in optimizer.param_groups:
                param_group['lr'] = max(param_group['lr'] * 0.9, 1e-6)
        
        state = env.reset()
        w_numpy = w.detach().numpy()
        v_numpy = v.detach().numpy()

        action = epsilon_greedy(lambda s, a: v_numpy@torch.relu(torch.from_numpy(w_numpy@feature_vector_nn(s, a, d))).numpy(), state, nA, eps)
        total_reward = 0

        while True:
            if action >= env.nA:
                print("Something went wrong")
                print(f"action attemped = {action}")
                break
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            if not done:
                w_numpy = w.detach().numpy()
                v_numpy = v.detach().numpy()

                next_action = epsilon_greedy(lambda s, a: v_numpy@torch.relu(torch.from_numpy(w_numpy@feature_vector_nn(s, a, d))).numpy(), next_state, nA, eps)

                x = torch.from_numpy(feature_vector_nn(state, action, d)).float()
                x_prime = torch.from_numpy(feature_vector_nn(next_state, next_action, d)).float()

                z = torch.relu(torch.matmul(w,x))
                z_prime = torch.relu(torch.matmul(w,x_prime))

                y = torch.matmul(v, z)
                y_prime = torch.matmul(v, z_prime)
                
                td_error = reward + gamma*y_prime - y

                optimizer.zero_grad()

                loss = td_error
                loss.backward(torch.ones_like(td_error))

                optimizer.step()

                state = next_state
                action = next_action
            if done:
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
    return w, v

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