import gym

import math
import random
import matplotlib

import matplotlib.pyplot as plt

from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

env = gym.make("CartPole-v1")

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
print(f"{device=}")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, transition: Transition):
        """Save a transition"""
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

n_actions = env.action_space.n
# reset returns state, info tuple
n_observations = len(env.reset()[0])
print(f"{n_actions=} {n_observations=}")

policy_model = DQN(n_observations=n_observations, n_actions=n_actions).to(device)
target_model = DQN(n_observations=n_observations, n_actions=n_actions).to(device)
target_model.load_state_dict(policy_model.state_dict())

LEARNING_RATE = 1e-4
# NOTE: own Adam, not AdamW
optimizer = optim.Adam(policy_model.parameters(), lr=LEARNING_RATE, amsgrad=True)


"""
Epsilon/ getting action
"""
epsilon = 0.95
EPSILON_DECAY = 0.99975
EPSILON_MIN = 0.01

#steps_done = 0

def select_action(state):
    # NOTE: own epsilon decay

    # TODO: avoid dupe epsilon decay
    if random.random() < epsilon:
        epsilon = max(EPSILON_MIN, epsilon*EPSILON_DECAY)
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
    else:
        epsilon = max(EPSILON_MIN, epsilon*EPSILON_DECAY)
        # IDK wtf this is, TODO
        shit = policy_model(state).max(1).indices.view(1,1)
        return shit


"""
Raw copy pasted
"""
episode_durations = []

def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    # if is_ipython:
    #     if not show_result:
    #         display.display(plt.gcf())
    #         display.clear_output(wait=True)
    #     else:
    #         display.display(plt.gcf())

"""
optimization
"""

memory = ReplayMemory(10000)
BATCH_SIZE = 128
GAMMA = 0.99


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    
    transitions = memory.sample(BATCH_SIZE)

    # from AoS to SoA
    batch = Transition(*zip(*transitions))

    # TODO: tweak, explore this
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
    # TODO: wtf is gather
    state_action_values = policy_model(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_model(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch


    # TODO: what is the size of these tensors?
    criterion = nn.SmoothL1Loss()
    # TODO: unsqueeze?
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    # basically, dont make the gradient too big
    torch.nn.utils.clip_grad_value_(policy_model.parameters(), 100)
    optimizer.step()
