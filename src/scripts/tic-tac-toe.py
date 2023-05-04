import random
import click
import heapq


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


AI_STRENGTH = -1

FIELD_EMPTY = 0
PLAYER_X = 1
PLAYER_O = -1

X_AXIS_LABELS = ['A', 'B', 'C']
Y_AXIS_LABELS = ['1', '2', '3']


def cyan(s):
    return f'{CliColors.CYAN}{s}{CliColors.END}'


def green(s):
    return f'{CliColors.GREEN}{s}{CliColors.END}'


def print_grid(_grid):
    def convert(num):
        if num == PLAYER_X:
            return 'X'
        if num == PLAYER_O:
            return 'O'
        return '_'

    print(f'\n    | {cyan(X_AXIS_LABELS[0])} | {cyan(X_AXIS_LABELS[1])} | {cyan(X_AXIS_LABELS[2])} |')

    for i in range(3):
        print(f'| {cyan(Y_AXIS_LABELS[i])} |', end='')
        for j in range(3):
            print(f' {convert(_grid[(i * 3) + j])} |', end='')
        print()


def get_input_coordinates(_input):
    _input = _input.upper().replace(' ', '')
    input_x = X_AXIS_LABELS.index(list(_input)[0])
    input_y = Y_AXIS_LABELS.index(list(_input)[1])
    return input_x, input_y


def get_action_coordinates_string(a):
    idx = a[1]
    return f'{X_AXIS_LABELS[int(idx % 3)]}{Y_AXIS_LABELS[int(idx / 3)]}'


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

    # print('\nAI strength:', AI_STRENGTH)
    action_rating = 1 if random.randint(1, 10) <= AI_STRENGTH else 2  # allow chance for mistakes (eg. strength 7 = 30%)
    # print('\nMove rating:', action_rating)
    return heapq.nsmallest(action_rating, sorted_list, key=lambda l: l[1])[-1]  # return action


# Initializing the state
grid = [FIELD_EMPTY for _ in range(9)]
print('|---------- { TIC TAC TOE } ----------|')
AI_OPPONENT = click.confirm(
    'Do you want to play against AI? Otherwise the opponents moves will be random.',
    default=True
)
if AI_OPPONENT:
    while AI_STRENGTH not in range(0, 11):
        AI_STRENGTH = int(input('How strong do you want your AI opponent to be? [0 = worst, 10 = best]: '))
print('You\'re X while your Opponent is O')

print_grid(grid)

# Run program while game is not terminated
while terminal(grid) is None:
    if player_turn(grid) == PLAYER_X:
        print('\n\nIt\'s your turn', end='\n')
        try:
            x, y = get_input_coordinates(input(f'Enter a coordinate [e.g. {X_AXIS_LABELS[0]}{Y_AXIS_LABELS[1]}]: '))
        except ValueError:
            print('That is not a valid coordinate. Please try again.')
            continue
        index = 3 * y + x

        if not grid[index] == FIELD_EMPTY:
            print('That coordinate is already taken. Please try again.')
            continue

        # Apply action and print grid
        grid = result(grid, (PLAYER_X, index))
        print_grid(grid)
    else:
        print('\n\nThe opponent is playing its turn: ', end='')
        o_action = None
        if AI_OPPONENT:
            # Get action by running the minimax algorithm
            o_action = minimax(grid)[0]
        else:
            empty_fields = [i for i in range(len(grid)) if grid[i] == FIELD_EMPTY]
            index = empty_fields[random.randint(0, len(empty_fields) - 1)]  # random index of empty grid-field
            o_action = (PLAYER_O, index)

        print(green(get_action_coordinates_string(o_action)))
        # Apply returned action to the state
        grid = result(grid, o_action)
        print_grid(grid)

# print winner
winner = terminal(grid)
if winner == PLAYER_X:
    print('You have won!')
elif winner == PLAYER_O:
    print('You have lost!')
else:
    print('It\'s a tie.')
