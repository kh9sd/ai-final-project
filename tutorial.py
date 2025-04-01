import gymnasium as gym

import math
import random
import matplotlib

import matplotlib.pyplot as plt

from collections import namedtuple, deque
import itertools

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

# Our goal is to make a NN that will
# state -> (Q(s,a_1), ..., Q(s,a_n))
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
    global epsilon

    # TODO: avoid dupe epsilon decay
    if random.random() < epsilon:
        epsilon = max(EPSILON_MIN, epsilon*EPSILON_DECAY)
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
    else:
        epsilon = max(EPSILON_MIN, epsilon*EPSILON_DECAY)
        """
        # ex: tensor([[0.1983, 0.1383]])
        """
        shit = policy_model(state)
        #print(f"policy_model(state) {shit=}")
        
        """
        # ex: torch.return_types.max(
        # values=tensor([0.1983], device='cuda:0', grad_fn=<MaxBackward0>),
        # indices=tensor([0], device='cuda:0'))
        # indices are good enough for argmax
        """
        # TLDR: this is our argmax
        shit = shit.max(1)
        #print(f"max(1) {shit=}")

        """
        # view dimensions are (1,1)
        """
        shit = shit.indices.view(1,1)
        
        """
        # ex: tensor([[0]])
        """
        #print(f"After it all: {shit=}")
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

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    """
    # non_final_mask is a Tensor of Bool, size is Size([128])
    """                                        
    # print(f"{non_final_mask=}, {non_final_mask.size()}")

    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])

    """
    # non_final_next_states is a tensor of states, filtered out None next states
    # ex:
    #  [ 5.8431e-02, -3.6820e-02, -6.3966e-02, -9.0537e-02],
    #  [ 2.7936e-02, -3.8984e-02, -5.0224e-03, -4.2624e-02],
    #  [-1.4962e-01, -4.4439e-01,  1.9825e-01,  8.9324e-01]], device='cuda:0'), torch.Size([124, 4])
    """
    # print(f"{non_final_next_states=}, {non_final_next_states.size()}")
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    
    policy_on_state_batch = policy_model(state_batch)
    """
    # note: n_actions=np.int64(2) n_observations=4

    # state_batch.size()=torch.Size([128, 4]) 
    # policy_on_state_batch.size()=torch.Size([128, 2]) 

    # action_batch.size()=torch.Size([128, 1])    
    This one is like
    [1],
    [1],
    [1],
    [1],
    [0],
    [0],

    # state_action_values.size()=torch.Size([128, 1])
    This one is like
    [-3.4153e-03],
    [ 2.3756e-01],
    [ 2.4197e-01],
    [-2.7275e-02],
    [ 2.3714e-01],
    [ 3.0386e-04],
    [ 2.8522e-01],
    [ 2.8025e-01]]
    """
    # TODO: how does the gather work here, what does it do?

    """
    wtf is gather?: 

    multi-index selection method

    >>> t = torch.tensor([[1, 2], 
                          [3, 4]])
    >>> torch.gather(t, 1, torch.tensor([[0, 0], [1, 0]]))
    tensor([[ 1,  1],
            [ 4,  3]])
    
    we are selecting elements from input

    1 is dimension, we want to collect from second? dimension

    indices for second dimension are [0,0], [1,0]


    we skip the first dimension, so the result's 1st dim is the same as the index's first dimension???

    indices hold second dimensions/column indices, not row indices
        output will have in it's first row, a selection of elements from input's first row

        for [0,0], we select the first elemnt of the first row of the input twice, [1,1] result


    I HAVE NO FUCKING CLUE

    same as
    current_Q_values = Q(obs_batch).gather(1, act_batch.unsqueeze(1))

    q_vals = []
    for qv, ac in zip(Q(obs_batch), act_batch):
        q_vals.append(qv[ac])
    q_vals = torch.cat(q_vals, dim=0)
    """
    # print(f"{policy_on_state_batch=}, \n {action_batch=}")

    # these are the Q(s,a) values
    """
    Basically, what is happening is that the model spits out
    [Q(s,a_1),
     Q(s,a_2),
     Q(s,a_3),
     ...
     Q(s,a_n)]
    in policy_on_state_batch.

    The actual action would've been an index into that tensor, 0 through n-1
    """
    state_action_values = policy_on_state_batch.gather(1, action_batch)

    # print(f"{state_batch.size()=} \
    #       {policy_on_state_batch.size()=} \
    #       {action_batch.size()=} \
    #       {state_action_values=}")

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_model(non_final_next_states).max(1).values
    # Compute the expected Q values
    # this is just the (r + gamma max_{a'}(Q(s',a'))) term!
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch


    criterion = nn.SmoothL1Loss()
    """
    unsqueeze:

    adds dim of size 1 at specified pos

    >>> x = torch.tensor([1, 2, 3, 4])
    size is [4]

    >>> torch.unsqueeze(x, 0)
    tensor([[ 1,  2,  3,  4]])
    size is [1,4]

    >>> torch.unsqueeze(x, 1)
    tensor([[ 1],
            [ 2],
            [ 3],
            [ 4]])
    size is [4,1]
    """
    # print(f"{state_action_values.size()=} {expected_state_action_values.size()=}")
    """
    # state_action_values.size()=torch.Size([128, 1])
    # expected_state_action_values.size()=torch.Size([128])
    """
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    # basically, dont make the gradient too big
    torch.nn.utils.clip_grad_value_(policy_model.parameters(), 100)
    optimizer.step()




"""
Core loop
"""

TARGET_MODEL_UPDATE_RATE = 0.005

if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 600
else:
    num_episodes = 50

print(f"Starting, {num_episodes=}")

for i_episode in range(num_episodes):
    print(f"{i_episode=}")
    # Initialize the environment and get its state
    state, info = env.reset()
    """
    # state is np array,
    # state=array([0.01756821, 0.03350502, 0.02066539, 0.04153426], dtype=float32)
    """
    #print(f"{state=}")

    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    
    # count is an infinite generator
    for t in itertools.count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        """
        # state.size()=torch.Size([1, 4]) 
        # action.size()=torch.Size([1, 1]) 
        # next_state.size()=torch.Size([1, 4]) 
        # reward.size()=torch.Size([1])
        """
        # print(f"{state.size()=} {action.size()=} {next_state.size()=} {reward.size()=}")

        """
        # state=tensor([[-0.0262, -0.4380,  0.1120,  0.7674]], device='cuda:0') 
        # action=tensor([[0]], device='cuda:0') 
        # next_state=tensor([[-0.0349, -0.6345,  0.1274,  1.0932]], device='cuda:0') 
        # reward=tensor([1.], device='cuda:0')
        """
        # print(f"{state=} {action=} {next_state=} {reward=}")

        # Store the transition in memory
        memory.push(Transition(state, action, next_state, reward))

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # This is the target updating
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_model.state_dict()
        policy_net_state_dict = policy_model.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TARGET_MODEL_UPDATE_RATE + target_net_state_dict[key]*(1-TARGET_MODEL_UPDATE_RATE)
        target_model.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break

print('Complete')

torch.save(policy_model.state_dict(), 'tutorial_model.h5')
plot_durations(show_result=True)
# plt.ioff()
plt.show()
