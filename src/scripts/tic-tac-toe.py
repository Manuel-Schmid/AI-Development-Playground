import random


class CliColors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


FIELD_EMPTY = 0
PLAYER_X = 1
PLAYER_O = -1

X_AXIS_LABELS = ['A', 'B', 'C']
Y_AXIS_LABELS = ['1', '2', '3']


def cyan(s):
    return f'{CliColors.CYAN}{s}{CliColors.END}'


def print_grid(_grid):
    def convert(num):
        if num == PLAYER_X:
            return 'X'
        if num == PLAYER_O:
            return 'O'
        return '_'

    print(f'\n    | {cyan("A")} | {cyan("B")} | {cyan("C")} |')

    for i in range(3):
        print(f'| {cyan(Y_AXIS_LABELS[i])} |', end='')
        for j in range(3):
            print(f' {convert(_grid[(i * 3) + j])} |', end='')
        print()


def player_turn(_grid):
    x_places = _grid.count(PLAYER_X)
    o_places = _grid.count(PLAYER_O)

    if x_places + o_places == 9:
        return None
    elif x_places > o_places:
        return PLAYER_O
    else:
        return PLAYER_X


def actions(_grid):
    player = player_turn(_grid)
    actions_list = [(player, i) for i in range(len(_grid)) if _grid[i] == FIELD_EMPTY]
    return actions_list


def result(_grid, action):
    (player, idx) = action
    grid_copy = _grid.copy()
    grid_copy[idx] = player
    return grid_copy


def terminal(_grid):  # is the game finished?
    for i in range(3):
        # Check rows
        if _grid[3 * i] == _grid[3 * i + 1] == _grid[3 * i + 2] != FIELD_EMPTY:
            return _grid[3 * i]
        # Check columns
        if _grid[i] == _grid[i + 3] == _grid[i + 6] != FIELD_EMPTY:
            return _grid[i]

    # Check diagonals
    if _grid[0] == _grid[4] == _grid[8] != FIELD_EMPTY:
        return _grid[0]
    if _grid[2] == _grid[4] == _grid[6] != FIELD_EMPTY:
        return _grid[2]

    # Check if moves are available
    if player_turn(_grid) is None:
        return 0

    # Game is not finished
    return None


def utility(_grid, cost):
    term = terminal(_grid)
    if term is not None:
        return term, cost  # cost of reaching terminal state

    action_list = actions(_grid)
    utils = []
    for action in action_list:
        n_grid = result(_grid, action)
        utils.append(utility(n_grid, cost + 1))  # Every recursion increments cost (depth)

    # Remember associated cost with score of state
    score = utils[0][0]
    idx_cost = utils[0][1]
    if player_turn(_grid) == PLAYER_X:
        for i in range(len(utils)):
            if utils[i][0] > score:
                score = utils[i][0]
                idx_cost = utils[i][1]
    else:
        for i in range(len(utils)):
            if utils[i][0] < score:
                score = utils[i][0]
                idx_cost = utils[i][1]

    return score, idx_cost  # score & associated cost


def minimax(_grid):
    action_list = actions(_grid)
    utils = []
    # Each item in utils contains action associated, score and cost of that action
    for action in action_list:
        n_grid = result(_grid, action)
        utils.append((action, utility(n_grid, 1)))

    # if utils has no objects return default action and utility
    if len(utils) == 0:
        return (0, 0), (0, 0)

    # Sort list in ascending order of cost
    sorted_list = sorted(utils, key=lambda l: l[0][1])

    action = min(sorted_list, key=lambda l: l[1])
    return action


# Initializing the state
grid = [FIELD_EMPTY for _ in range(9)]
print('|-------- { TIC TAC TOE } --------|')
print('You\'re X while the Computer is O')

print_grid(grid)

# Run program while game is not terminated
while terminal(grid) is None:
    if player_turn(grid) == PLAYER_X:
        print('\n\nIt\'s your turn', end='\n')
        x = int(X_AXIS_LABELS.index(input('Enter the x-coordinate [A-C]: ')))
        y = int(input('Enter the y-coordinate [1-3]: ')) - 1
        index = 3 * y + x

        if not grid[index] == FIELD_EMPTY:
            print('That coordinate is already taken. Please try again.')
            continue

        # Apply action and print grid
        grid = result(grid, (PLAYER_X, index))
        print_grid(grid)
    else:
        # random index as temporary replacement for computer
        empty_fields = [i for i in range(len(grid)) if grid[i] == FIELD_EMPTY]
        index = empty_fields[random.randint(0, len(empty_fields) - 1)]
        # Apply action and print grid
        grid = result(grid, (PLAYER_O, index))
        print_grid(grid)

# print winner
winner = terminal(grid)
if winner == PLAYER_X:
    print('You have won!')
elif winner == PLAYER_O:
    print('You have lost!')
else:
    print('It\'s a tie.')


