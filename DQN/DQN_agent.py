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


import torch
from torch.nn import Conv2d, ReLU, Flatten, Linear, Sequential, LazyLinear
import torch.nn as nn
import torch.optim as optim

torch.set_default_device('cuda')

def create_dqn(input_dims, n_actions, conv_units, dense_units):
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
# TODO: not convinced learn rate changes
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

def NHWC_to_NHWC(tensor):
    return tensor.permute(0, 3, 1, 2) # from NHWC to NCHW


def numpy_state_batch_to_tensor_state_batch(numpy_arr):
    shit_tensor = NHWC_to_NHWC(torch.tensor(numpy_arr))
    #shit_tensor = NHWC_to_NHWC(torch.tensor(reshaped_state, dtype=torch.half))
    # print(f"{shit_tensor=}")
    shit_tensor = shit_tensor.type(torch.cuda.FloatTensor)
    return shit_tensor

class DQNAgent(object):
    def __init__(self, env, model_name=MODEL_NAME, conv_units=64, dense_units=256):
        self.env = env

        # Deep Q-learning Parameters
        self.discount = DISCOUNT
        self.learn_rate = learn_rate
        self.epsilon = epsilon
        self.model = create_dqn(
            self.env.state_im.shape, self.env.ntiles, conv_units, dense_units)

        self.optimizer = optim.Adam(self.model.parameters(), lr=learn_rate, eps=1e-4)
        self.criterion = nn.MSELoss()

        # target model - this is what we predict against every step
        self.target_model = create_dqn(
            self.env.state_im.shape, self.env.ntiles, conv_units, dense_units)
    
        #self.target_model.set_weights(self.model.get_weights())
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.replay_memory = deque(maxlen=MEM_SIZE)
        self.target_update_counter = 0

        # self.tensorboard = ModifiedTensorBoard(
        #     log_dir=f'logs\\{model_name}', profile_batch=0)

    def get_action(self, state):
        self.model.eval()

        with torch.no_grad(): 
            board = state.reshape(1, self.env.ntiles)
            # TODO: verify this works
            unsolved = [i for i, x in enumerate(board[0]) if x==-0.125]

            rand = np.random.random() # random value b/w 0 & 1

            # TODO: print epsilon
            if rand < self.epsilon: # random move (explore)
                move = np.random.choice(unsolved)
            else:
                # TODO: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
                #moves = self.model.predict(np.reshape(state, (1, self.env.nrows, self.env.ncols, 1)))
                # pytorch expects of 
                reshaped_state = np.reshape(state, (1, self.env.nrows, self.env.ncols, 1))
                # print(f"{reshaped_state.shape=}")
                shit_tensor = numpy_state_batch_to_tensor_state_batch(reshaped_state)

                moves = self.model(shit_tensor)

                # convert back to numpy
                moves = moves.detach().cpu().numpy()
                moves[board!=-0.125] = np.min(moves) # set already clicked tiles to min value
                move = np.argmax(moves)

        return move

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self, done: bool):
        self.model.eval() 

        if len(self.replay_memory) < MEM_SIZE_MIN:
            return

        batch = random.sample(self.replay_memory, BATCH_SIZE)

        # TODO: print out data type of transition
        # might just be s,a,s',r?
        # nope, s,a,r,s',done
        # TODO: eval?
        current_states = np.array([transition[0] for transition in batch])
        current_states = numpy_state_batch_to_tensor_state_batch(current_states)
        current_qs_list = self.model(current_states)

        new_current_states = np.array([transition[3] for transition in batch])
        new_current_states = numpy_state_batch_to_tensor_state_batch(new_current_states)
        future_qs_list = self.target_model(new_current_states)

        # list of numpy arrays, list of tensors
        inputs, expected_outputs = [], []

        for i, (current_state, action, reward, new_current_state, done) in enumerate(batch):
            if not done:
                # this is where we get the shit from the target_model
                # print(f"{future_qs_list[i]=}")
                # TODO: is the item a good idea?
                max_future_q = torch.max(future_qs_list[i]).item()
                # print(f"{max_future_q=}")
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[i]
            # todo: are we not supposed to add the shit?
            current_qs[action] = new_q

            inputs.append(current_state)
            expected_outputs.append(current_qs)

        self.model.train() 
        #print(f"{len(inputs)=} {inputs[0].Size()}")
        inputs = numpy_state_batch_to_tensor_state_batch(inputs)
        # TODO: I don't think we care about BATCH SIZE, shuffle is taken care of already
        # print(expected_outputs)
        expected_outputs = torch.stack(expected_outputs, dim=0)
        # self.model.fit(np.array(X), np.array(y), batch_size=BATCH_SIZE,
        #                shuffle=False, verbose=0, callbacks=[self.tensorboard]\
        #                if done else None)
        self.optimizer.zero_grad()
        # print(f"{inputs.size()=}, {expected_outputs.size()=}")

        actual_outputs = self.model(inputs)

        loss = self.criterion(actual_outputs, expected_outputs)
        loss.backward()
        self.optimizer.step()

        # updating to determine if we want to update target_model yet
        if done:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            # self.target_model.set_weights(self.model.get_weights())
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_update_counter = 0

        # decay learn_rate
        self.learn_rate = max(LEARN_MIN, self.learn_rate*LEARN_DECAY)

        # decay epsilon
        self.epsilon = max(EPSILON_MIN, self.epsilon*EPSILON_DECAY)

# TODO: try debugging with 5 x 5 grid?
# TODO: just try it again?
if __name__ == "__main__":
    DQNAgent(MinesweeperEnv(9,9,10))
