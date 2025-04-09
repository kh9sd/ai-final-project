
from minesweeper_env import MinesweeperEnv
import tkinter
import matplotlib.pyplot as plt
import numpy as np
import random

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

    action = random.randint(0, 35)

    new_state, _, new_done = env.step(action)

    state, done = new_state, new_done

# display ending move
display_minesweeper_window(env)
