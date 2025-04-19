from minesweeper_env import MinesweeperEnv

import tqdm
import random
import torch
from minesweeper import DQN, MINESWEEPER_HEIGHT, MINESWEEPER_WIDTH, MINESWEEPER_N_MINES

env = MinesweeperEnv(width=MINESWEEPER_WIDTH, height=MINESWEEPER_HEIGHT, n_mines=MINESWEEPER_N_MINES)

print("playing new game")

n_actions = env.ntiles

policy_model = DQN(n_actions = n_actions)
"""
On 6x6, with 6 mines
Final win rate: 24/10000 = 0.0024

On n_mines = random.randint(4,8)
Final win rate: 111/10000 = 0.0111
"""
# Random

"""
On 6x6, with 6 mines
Final win rate: 4844/10000 = 0.4844

On n_mines = random.randint(4,8)
Final win rate: 4662/10000 = 0.4662
"""
policy_model.load_state_dict(torch.load('models/Apr12_12-54-22_DESKTOP-L4QOK81_minesweeper/800000.h5'))
#policy_model.load_state_dict(torch.load('models/Apr07_12-08-41_DESKTOP-L4QOK81_minesweeper/260000.h5'))
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

def select_action_random(state):
    # print(f"{state.size()=}")
    assert(state.size() == (1, 2, MINESWEEPER_HEIGHT, MINESWEEPER_WIDTH))
    # print(f"{state=}")

    flattened_board = state[0,1].reshape(1, env.ntiles)
    #print(f"{flattened_board=}")
    # expect to be 1 if unsolved
    unsolved_mask = torch.ones(flattened_board.shape, dtype=torch.float32)

    unsolved_action_tensor = torch.isclose(flattened_board, unsolved_mask)
    #print(f"{unsolved_action_tensor=}")

    unsolved_action_indices = [i for i, x in enumerate(unsolved_action_tensor[0]) if x == True]

    return torch.tensor([[random.choice(unsolved_action_indices)]], dtype=torch.long)


def env_state_to_tensor_batch_state(state):
    assert(state.shape == (2, MINESWEEPER_HEIGHT, MINESWEEPER_WIDTH))

    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    assert(state.size() == (1, 2, MINESWEEPER_HEIGHT, MINESWEEPER_WIDTH))

    assert(state.size() == (1, 2, MINESWEEPER_HEIGHT, MINESWEEPER_WIDTH))

    return state


# returns 1 if won, 0 is lost
def run_game():
    n_mines = random.randint(4,8)
    # n_mines = MINESWEEPER_N_MINES

    env = MinesweeperEnv(width=MINESWEEPER_WIDTH, height=MINESWEEPER_HEIGHT, n_mines=n_mines)
    past_n_wins = env.n_wins
    state, done = env.reset(), False

    while not done:
        # Run with actual NN model
        # action = select_action(env_state_to_tensor_batch_state(state))

        # Run with random model
        action = select_action_random(env_state_to_tensor_batch_state(state))
        # print(f"{action=}")

        new_state, _, new_done = env.step(action)

        state, done = new_state, new_done
        # print(f"{state}")
    
    return env.n_wins - past_n_wins


AGG_STATS_EVERY = 100 # calculate stats every 100
GAMES_TO_PLAY = 10_000
games_won = 0

print(f"Running {GAMES_TO_PLAY} games for verification")

for i_episode in tqdm.tqdm(range(1, GAMES_TO_PLAY), unit='episode'):
    games_won += run_game()

    if (i_episode%AGG_STATS_EVERY == 0):
        print(f"Win rate: {games_won}/{i_episode}")

print(f"Final win rate: {games_won}/{GAMES_TO_PLAY} = {games_won/GAMES_TO_PLAY}")
