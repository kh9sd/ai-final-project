from minesweeper_env import MinesweeperEnv
import random

from collections import namedtuple, deque
import itertools

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

import numpy as np
import tqdm

MINESWEEPER_HEIGHT = 6
MINESWEEPER_WIDTH = 5
MINESWEEPER_N_MINES = 5

import pickle

env = MinesweeperEnv(width=MINESWEEPER_WIDTH, height=MINESWEEPER_HEIGHT, n_mines=MINESWEEPER_N_MINES)

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


CONV_FEATURES = 512
LINEAR_FEATURES = 512

class ConvRELu(nn.Module):
    def __init__(self, n_inputs):
        super(ConvRELu, self).__init__()
        self.layers = nn.Sequential(
                nn.Conv2d(n_inputs, CONV_FEATURES, kernel_size=3,  padding='same'),
                nn.ReLU(),
        )
    
    def forward(self, x):
        return self.layers(x)


class LazyConvRELu(nn.Module):
    def __init__(self):
        super(LazyConvRELu, self).__init__()
        self.layers = nn.Sequential(
                nn.LazyConv2d(CONV_FEATURES, kernel_size=3, padding='same'),
                nn.ReLU(),
        )
    
    def forward(self, x):
        return self.layers(x)

# Our goal is to make a NN that will
# state -> (Q(s,a_1), ..., Q(s,a_n))
class DQN(nn.Module):
    def __init__(self, n_actions):
        super(DQN, self).__init__()

        self.first_conv_layer = ConvRELu(1)
        
        self.second_conv_layer = ConvRELu(512)
        self.third_conv_layer = ConvRELu(512)
        self.fourth_conv_layer = ConvRELu(512)
        # self.second_conv_layer = LazyConvRELu()
        # self.third_conv_layer = LazyConvRELu()
        # self.fourth_conv_layer = LazyConvRELu()

        self.flatten = nn.Flatten()

        self.first_linear = nn.Linear(512 * env.ntiles, LINEAR_FEATURES)
        #self.first_linear = nn.LazyLinear(LINEAR_FEATURES)

        self.remaining_layers = nn.Sequential(
                nn.ReLU(),
                nn.Linear(LINEAR_FEATURES, LINEAR_FEATURES),
                nn.ReLU(),
                nn.Linear(LINEAR_FEATURES, n_actions))

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        after_first = self.first_conv_layer(x)
        after_second = self.second_conv_layer(after_first)
        after_third = self.third_conv_layer(after_second)
        after_fourth = self.fourth_conv_layer(after_third)

        #print(f"{after_first.size()=} \n{after_second.size()=} \n{after_third.size()=} \n{after_fourth.size()=}")

        flattened = self.flatten(after_fourth)
        #print(f"{flattened.size()=}")

        after_first_linear = self.first_linear(flattened)

        return self.remaining_layers(after_first_linear)

n_actions = env.ntiles
# reset returns state, info tuple
#n_observations = len(env.reset()[0])
print(f"{n_actions=}")

policy_model = DQN(n_actions=n_actions).to(device)
target_model = DQN(n_actions=n_actions).to(device)
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

def NHWC_to_NHWC(tensor):
    return tensor.permute(0, 3, 1, 2) # from NHWC to NCHW

# Returns tensor of size =torch.Size([1, 1])
def select_action(state):
    # print(f"{state.size()=}")
    assert(state.size() == (1, 1, MINESWEEPER_HEIGHT, MINESWEEPER_WIDTH))
    # print(f"{state=}")

    # NOTE: own epsilon decay
    global epsilon
    epsilon = max(EPSILON_MIN, epsilon*EPSILON_DECAY)

    flattened_board = state.reshape(1, env.ntiles)
    # print(f"{flattened_board=}")

    #if 1 < epsilon:
    if random.random() < epsilon:
        # actions indices, filter out already solved tiles
        # TODO: make this not strict equality? floats?
        unsolved_action_indices = [i for i, x in enumerate(flattened_board[0]) if x==-0.125]
        # print(f"{unsolved_action_indices=}")

        return torch.tensor([[random.choice(unsolved_action_indices)]], device=device, dtype=torch.long)
    else:
        """
        # ex: tensor([[0.1983, 0.1383]])
        """
        moves = policy_model(state)
        assert(moves.size() == (1, n_actions))
        # print(f"policy_model(state) {moves=}")

        # moves[board!=-0.125] = np.min(moves) # set already clicked tiles to min value

        # The basic idea is to set the Q-value already picked actions to the very minimum
        # NOTE: changed to be even below minimum
        # TODO: this shit, wtf? strict equality
        shit_mask = flattened_board!=-0.125
        # print(f"{shit_mask=}")

        # print(f"{torch.min(moves)=}")
        moves[shit_mask] = torch.min(moves).item() - 1

        # print(f"{moves=}")
        
        """
        # ex: torch.return_types.max(
        # values=tensor([0.1983], device='cuda:0', grad_fn=<MaxBackward0>),
        # indices=tensor([0], device='cuda:0'))
        # indices are good enough for argmax
        """
        # TLDR: this is our argmax
        shit = moves.max(1)
        #print(f"max(1) {shit=}")

        """
        # view dimensions are (1,1)
        """
        shit = shit.indices.view(1,1)
        
        """
        # ex: tensor([[0]])
        """
        #print(f"After it all: {shit=}")

        assert(shit.size() == (1,1))
        return shit


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

    # Compute max_{a'}(Q(s',a'))) for all next states
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
AGG_STATS_EVERY = 100 # calculate stats every 100 games for tensorboard
SAVE_MODEL_EVERY = 10_000 # save model and replay every 10,000 episodes

writer = SummaryWriter(comment="_minesweeper")
log_dir_rel_path = writer.log_dir

TRAINING_NAME = log_dir_rel_path.split("\\")[1]
print(f"{TRAINING_NAME=}")

if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 100000
else:
    num_episodes = 50

print(f"Starting, {num_episodes=}")

def env_state_to_tensor_batch_state(state):
    assert(state.shape == (MINESWEEPER_HEIGHT, MINESWEEPER_WIDTH, 1))

    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    assert(state.size() == (1, MINESWEEPER_HEIGHT, MINESWEEPER_WIDTH, 1))

    state = NHWC_to_NHWC(state)
    # TODO: look at values, verify
    assert(state.size() == (1, 1, MINESWEEPER_HEIGHT, MINESWEEPER_WIDTH))

    return state

progress_list = []
episode_rewards = []
wins_list = []

#for i_episode in range(num_episodes):
for i_episode in tqdm.tqdm(range(num_episodes), unit='episode'):
    # print(f"{i_episode=}")
    # Initialize the environment and get its state
    state = env.reset()
    """
    # state is np array,
    # state=array([0.01756821, 0.03350502, 0.02066539, 0.04153426], dtype=float32)
    """
    # print(f"{state=} {state.shape=}")
    state = env_state_to_tensor_batch_state(state) 
    
    past_env_wins = env.n_wins
    episode_reward = 0

    # count is an infinite generator
    for t in itertools.count():
        # print(f"{t=}")
        action = select_action(state)
        observation, reward, done = env.step(action.item())
        episode_reward += reward

        reward = torch.tensor([reward], device=device)

        if done:
            next_state = None
        else:
            next_state = env_state_to_tensor_batch_state(observation)
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
            break
    
    # After a game, metrics
    progress_list.append(env.n_progress) # n of non-guess moves
    episode_rewards.append(episode_reward)
    wins_list.append(env.n_wins - past_env_wins)
    
    if (i_episode % AGG_STATS_EVERY == 0):
        median_progress = np.median(progress_list[-AGG_STATS_EVERY:])
        win_rate = np.sum(wins_list[-AGG_STATS_EVERY:]) / AGG_STATS_EVERY
        median_reward = np.median(episode_rewards[-AGG_STATS_EVERY:])

        writer.add_scalar("Median progress/train", median_progress, i_episode)
        writer.add_scalar("Win rate/train", win_rate, i_episode)
        writer.add_scalar("Median reward/train", median_reward, i_episode)
        writer.add_scalar("Epsilon", epsilon, i_episode)

        print(f'Episode: {i_episode}, Median progress: {median_progress}, Median reward: {median_reward}, Win rate : {win_rate}')
    
    if (i_episode % SAVE_MODEL_EVERY == 0):
        with open(f'replay/{TRAINING_NAME}_{i_episode}.pkl', 'wb') as output:
            pickle.dump(memory, output)

        torch.save(policy_model.state_dict(), f'models/{TRAINING_NAME}_{i_episode}.h5')

# TODO: need to save last iteration
print('Complete')
