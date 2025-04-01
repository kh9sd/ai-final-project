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

    def push(self, transition):
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
