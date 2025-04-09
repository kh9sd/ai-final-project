
from minesweeper_env import MinesweeperEnv
import tkinter
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
from minesweeper import DQN

MINESWEEPER_HEIGHT = 6
MINESWEEPER_WIDTH = 6
MINESWEEPER_N_MINES = 6

env = MinesweeperEnv(width=MINESWEEPER_WIDTH, height=MINESWEEPER_HEIGHT, n_mines=MINESWEEPER_N_MINES)

print("playing new game")


# state_im = [t['value'] for t in env.state]
# state_im = np.array(state_im, dtype=object)
# state_im = np.reshape(state_im, (MINESWEEPER_HEIGHT, MINESWEEPER_WIDTH))
# print(f"{state_im=}")

PIXELS_PER_SQUARE = 70

def get_color_and_number(value):
    if value == 'B':
        return "", "red"
    elif value == 'U':
        return "", "white"
    else:
        return value, "light gray"


state, done = env.reset(), False
print(f"{state=} {env.state=}")
n_actions = env.ntiles

policy_model = DQN(n_actions = n_actions)
policy_model.load_state_dict(torch.load('models/Apr07_12-08-41_DESKTOP-L4QOK81_minesweeper/260000.h5'))
policy_model.eval()

# Returns tensor of size =torch.Size([1, 1])
def select_action(state):
    # print(f"{state.size()=}")
    assert(state.size() == (1, 2, MINESWEEPER_HEIGHT, MINESWEEPER_WIDTH))
    # print(f"{state=}")

    flattened_board = state[0,1].reshape(1, env.ntiles)
    #print(f"{flattened_board=}")

    moves = policy_model(state)

    assert(moves.size() == (1, n_actions))
    # print(f"policy_model(state) {moves=}")

    # moves[board!=-0.125] = np.min(moves) # set already clicked tiles to min value

    # The basic idea is to set the Q-value already picked actions to the very minimum
    # that way we do not pick it
    # NOTE: changed to be even below minimum

    solved_mask = torch.zeros(flattened_board.shape, dtype=torch.float32)
    shit_mask = torch.isclose(flattened_board, solved_mask)
    # shit_mask = flattened_board!=-0.125
    #print(f"{shit_mask=}")

    #print(f"{torch.min(moves)=}")
    moves[shit_mask] = torch.min(moves).item() - 1

    #print(f"{moves=}")
    
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
    

def env_state_to_tensor_batch_state(state):
    assert(state.shape == (2, MINESWEEPER_HEIGHT, MINESWEEPER_WIDTH))

    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    assert(state.size() == (1, 2, MINESWEEPER_HEIGHT, MINESWEEPER_WIDTH))

    # TODO: look at values, verify
    assert(state.size() == (1, 2, MINESWEEPER_HEIGHT, MINESWEEPER_WIDTH))

    return state

def display_minesweeper_window(env):
    window = tkinter.Tk()
    window.minsize(width=PIXELS_PER_SQUARE*MINESWEEPER_WIDTH, height=PIXELS_PER_SQUARE*MINESWEEPER_HEIGHT)

    for state_coord_info in env.state:
        # each is like {'coord': (0, 0), 'value': 'U'}
        x, y = state_coord_info['coord']
        value = state_coord_info['value']
        print(f"{x=} {y=} {value=}")

        text, color = get_color_and_number(value)

        e = tkinter.Label(window, highlightbackground="black", highlightthickness=2, borderwidth=2, font=("Courier", 44), text=text, background=color)
        # TODO: verify, are x and y swapped?
        e.grid(row=x, column=y, sticky='nwse')

    window.grid_columnconfigure(list(range(6)), minsize=PIXELS_PER_SQUARE)
    window.grid_rowconfigure(list(range(6)), minsize=PIXELS_PER_SQUARE)

    window.mainloop()

while not done:
    display_minesweeper_window(env)

    action = select_action(env_state_to_tensor_batch_state(state))
    # print(f"{action=}")

    new_state, _, new_done = env.step(action)

    state, done = new_state, new_done

# display ending move
display_minesweeper_window(env)
