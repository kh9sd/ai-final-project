import os, sys

ROOT = os.getcwd()
print(ROOT)
sys.path.insert(1, f'{os.path.dirname(ROOT)}')

import warnings
warnings.filterwarnings('ignore')

from collections import deque
from minesweeper_env import *
# use my_tensorboard2.py if using tensorflow v2+, use my_tensorboard.py otherwise
# from my_tensorboard2 import *
#from DQN import *


from torch.nn import Conv2d, ReLU, Flatten, Linear, Sequential, LazyLinear
import torch.nn as nn
import torch.optim as optim

def create_dqn(learn_rate, input_dims, n_actions, conv_units, dense_units):
    print(f"{input_dims=} {conv_units=}")
    model = Sequential(
                Conv2d(1, conv_units, kernel_size=3,  padding='same'),
                ReLU(),
                Conv2d(conv_units, conv_units, kernel_size=3, padding='same'),
                ReLU(),
                Conv2d(conv_units, conv_units, kernel_size=3, padding='same'),
                ReLU(),
                Conv2d(conv_units, conv_units, kernel_size=3, padding='same'),
                ReLU(),
                Flatten(),
                # TODO: def need to change this 1
                LazyLinear(dense_units),
                ReLU(),
                Linear(dense_units, dense_units),
                ReLU(),
                Linear(dense_units, n_actions))

    # model.compile(optimizer=Adam(lr=learn_rate, epsilon=1e-4), loss='mse')

    print(model)
    return model

# Environment settings
MEM_SIZE = 50_000 # number of moves to store in replay buffer
MEM_SIZE_MIN = 1_000 # min number of moves in replay buffer

# Learning settings
BATCH_SIZE = 64
learn_rate = 0.01
LEARN_DECAY = 0.99975
LEARN_MIN = 0.001
DISCOUNT = 0.1 #gamma

# Exploration settings
epsilon = 0.95
EPSILON_DECAY = 0.99975
EPSILON_MIN = 0.01

# DQN settings
CONV_UNITS = 64 # number of neurons in each conv layer
DENSE_UNITS = 512 # number of neurons in fully connected dense layer
UPDATE_TARGET_EVERY = 5

# Default model name
MODEL_NAME = f'conv{CONV_UNITS}x4_dense{DENSE_UNITS}x2_y{DISCOUNT}_minlr{LEARN_MIN}'

class DQNAgent(object):
    def __init__(self, env, model_name=MODEL_NAME, conv_units=64, dense_units=256):
        self.env = env

        # Deep Q-learning Parameters
        self.discount = DISCOUNT
        self.learn_rate = learn_rate
        self.epsilon = epsilon
        self.model = create_dqn(
            self.learn_rate, self.env.state_im.shape, self.env.ntiles, conv_units, dense_units)

        self.optimizer = optim.Adam(self.model.parameters(), lr=learn_rate, eps=1e-4)
        self.criterion = nn.MSELoss()

        # target model - this is what we predict against every step
        self.target_model = create_dqn(
            self.learn_rate, self.env.state_im.shape, self.env.ntiles, conv_units, dense_units)
    
        #self.target_model.set_weights(self.model.get_weights())
        self.target_model.load_state_dict(self.target_model.state_dict())

        self.replay_memory = deque(maxlen=MEM_SIZE)
        self.target_update_counter = 0

        # self.tensorboard = ModifiedTensorBoard(
        #     log_dir=f'logs\\{model_name}', profile_batch=0)

    def get_action(self, state):
        board = state.reshape(1, self.env.ntiles)
        unsolved = [i for i, x in enumerate(board[0]) if x==-0.125]

        rand = np.random.random() # random value b/w 0 & 1

        if rand < self.epsilon: # random move (explore)
            move = np.random.choice(unsolved)
        else:
            #moves = self.model.predict(np.reshape(state, (1, self.env.nrows, self.env.ncols, 1)))
            moves = self.model(np.reshape(state, (1, self.env.nrows, self.env.ncols, 1)))
            moves[board!=-0.125] = np.min(moves) # set already clicked tiles to min value
            move = np.argmax(moves)

        return move

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self, done: bool):
        if len(self.replay_memory) < MEM_SIZE_MIN:
            return

        batch = random.sample(self.replay_memory, BATCH_SIZE)

        current_states = np.array([transition[0] for transition in batch])
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in batch])
        future_qs_list = self.target_model.predict(new_current_states)

        inputs, expected_outputs = [], []

        for i, (current_state, action, reward, new_current_state, done) in enumerate(batch):
            if not done:
                # this is where we get the shit from the target_model
                max_future_q = np.max(future_qs_list[i])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[i]
            current_qs[action] = new_q

            inputs.append(current_state)
            expected_outputs.append(current_qs)

        inputs = np.array(inputs)
        # TODO: I don't think we care about BATCH SIZE, shuffle is taken care of already
        expected_outputs = np.array(expected_outputs)
        # self.model.fit(np.array(X), np.array(y), batch_size=BATCH_SIZE,
        #                shuffle=False, verbose=0, callbacks=[self.tensorboard]\
        #                if done else None)
        self.optimizer.zero_grad()

        actual_outputs = self.model(inputs)

        loss = self.criterion(actual_outputs, expected_outputs)
        loss.backward()
        self.optimizer.step()

        # updating to determine if we want to update target_model yet
        if done:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

        # decay learn_rate
        self.learn_rate = max(LEARN_MIN, self.learn_rate*LEARN_DECAY)

        # decay epsilon
        self.epsilon = max(EPSILON_MIN, self.epsilon*EPSILON_DECAY)

if __name__ == "__main__":
    DQNAgent(MinesweeperEnv(9,9,10))
