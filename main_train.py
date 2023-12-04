from environment import *
import numpy as np
import pandas as pd
import torch
from torch.distributions import Categorical
from torch import nn, optim
import random
import matplotlib
import matplotlib.pyplot as plt
from gymnasium.wrappers import FlattenObservation

from collections import namedtuple, deque
from itertools import count, chain

# import csv

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")  
print("device: ", device)

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

RAALGO = 'slottedaloha'

forwardprobability = 0.5
writing = 1
p_sred = 0
p_max = 0.15
totaltime = 0
maxrep = 1
df = pd.DataFrame()

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

env = ShowerEnv()
# env = ShowerEnv()
n_actions = env.action_space.n
state, info = env.reset()
n_observation = len(state)

policy_net = DQN(n_observation, n_actions).to(device)
target_net = DQN(n_observation, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=0.001, amsgrad=True)
BATCH_SIZE = 128
memory = ReplayMemory(BATCH_SIZE)

steps_done = 0
episode_rewards = []

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = 0.1
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # Return the action with the largest expected reward
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
    
def plot_rewards(show_result=False):
    plt.figure(1)
    reward_t = torch.tensor(episode_rewards, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title(f'Training..., RA: {RAALGO}, Nodes: {NUMNODES}')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(reward_t.numpy())
    # Take 10 episode averages and plot them too
    if len(reward_t) >= 10:
        means = reward_t.unfold(0, 10, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(9), means))
        plt.plot(means.numpy())
    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
        
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * 0.99) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    
num_episodes = 1000

for i_episode in range(num_episodes):
    # Initialize the environment and state
    dflog = ra.randomaccess(NUMNODES, BEACONINTERVAL, FRAMETXSLOT, PER, RAALGO)
    dflog = dflog[dflog['result'] == 'succ']
    dflog = dflog.reset_index(drop=True)
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    rewards = 0
        
    for epoch in range(BEACONINTERVAL//TIMEEPOCH):
        # Select and perform an action
        action = select_action(state)
        # print(info)
        env.probenqueue(dflog)
        # print(info)
        observation, reward, terminated, truncated, info = env.step(action.item())
        # observation, reward, terminated, truncated, info = env.step_rlaqm(action.item(), dflog)
        reward = torch.tensor([reward], device=device)
        print(f"Iter: {i_episode}, Epoch: {epoch}, Action: {action.item()}, Reward: {reward.item()}")
        
        done = terminated or truncated
        if done:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*0.005 + target_net_state_dict[key]*(1-0.005)
        target_net.load_state_dict(target_net_state_dict)
        
        # print(f"Episode: {i_episode}/{num_episodes}, Epoch: {epoch}/{BEACONINTERVAL//TIMEEPOCH}, Action: {action.item()}, Reward: {reward.item()}")
        
        rewards += reward.item()
        
        if done:
            episode_rewards.append(rewards)
            plot_rewards()
    
print(f'Complete, RA: {RAALGO}, Nodes: {NUMNODES}')
plot_rewards(show_result=True)
plt.ioff()

# Save plot
plt.savefig('result.png')

plt.show()

filename = f'policy_model_deepaaqm_{RAALGO}_{NUMNODES}'

if writing == 1:
    torch.save(policy_net, filename + '.pt')
    
# Save returns for each episode to csv file
df = pd.DataFrame(episode_rewards, columns=['reward'])
df.to_csv(filename + '.csv')