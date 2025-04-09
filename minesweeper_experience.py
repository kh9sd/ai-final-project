
from minesweeper_env import MinesweeperEnv
import tkinter
import matplotlib.pyplot as plt
import numpy as np

MINESWEEPER_HEIGHT = 6
MINESWEEPER_WIDTH = 6
MINESWEEPER_N_MINES = 6

env = MinesweeperEnv(width=MINESWEEPER_WIDTH, height=MINESWEEPER_HEIGHT, n_mines=MINESWEEPER_N_MINES)

print("playing new game")
state = env.reset()

print(f"{state=} {env.state=}")

state_im = [t['value'] for t in env.state]
state_im = np.array(state_im, dtype=object)
state_im = np.reshape(state_im, (MINESWEEPER_HEIGHT, MINESWEEPER_WIDTH))
print(f"{state_im=}")
# ax = plt.subplot(8, 4, i)
# plt.show()

# for i, output_kernel in enumerate(output, start=1):
#     # print(f"{output_kernel.size()=}")
#     output_kernel = output_kernel.detach().numpy()
#     ax = plt.subplot(8, 4, i)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     plt.imshow(output_kernel, cmap='gray')


# class UI:
#     def __init__(self, universe):
#         self.cells = [[]]
#         # self.universe = universe

#         window = tkinter.Tk()
#         window.minsize(width=500, height=500)

#         canvas = tkinter.Canvas(width=500, height=500)
#         self.fill_canvas(canvas)
#         canvas.pack()

#         button = tkinter.Button(text='go', command=self.button_clicked, width=5, height=2)
#         button.pack()

#         window.mainloop()

#     def button_clicked(self):
#         self.universe.update()
#         for i in range(self.universe.max_row):
#             for j in range(self.universe.max_col):
#                 self.cells[i][j].configure(background='white' if self.universe.board[i][j] == 0 else 'black')

#     def fill_canvas(self, canvas: tkinter.Canvas):
#         for i in range(self.universe.max_row):
#             self.cells.append([])
#             for j in range(self.universe.max_col):
#                 e = tkinter.Entry(canvas, width=2, background='white' if self.universe.board[i][j] == 0 else 'black')
#                 e.grid(row=i, column=j)
#                 self.cells[i].append(e)

PIXELS_PER_SQUARE = 70

def get_color_and_number(value):
    if value == 'B':
        return "", "red"
    elif value == 'U':
        return "", "white"
    else:
        return value, "light gray"


import random
while True:
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

    action = random.randint(0, 35)
    env.step(action)
