from minesweeper_env import MinesweeperEnv

"""
    parser.add_argument('--width', type=int, default=9,
                        help='width of the board')
    parser.add_argument('--height', type=int, default=9,
                        help='height of the board')
    parser.add_argument('--n_mines', type=int, default=10,
                        help='Number of mines on the board')
"""

WIDTH = 9
HEIGHT = 9
NUM_MINES = 10

shit = MinesweeperEnv(width=WIDTH, height=HEIGHT, n_mines=NUM_MINES)

# init_grid() is called in constructor already
# same for init_state() and get_board()
# get_state_im is board state getter for DQN

shit.draw_state(shit.get_state_im(shit.state))
