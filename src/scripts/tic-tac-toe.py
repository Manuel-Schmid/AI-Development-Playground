FIELD_EMPTY = 0
PLAYER_X = 1
PLAYER_O = -1

X_AXIS_LABELS = ["A", "B", "C"]
Y_AXIS_LABELS = ["1", "2", "3"]


def print_grid(_grid):
    def convert(num):
        if num == PLAYER_X:
            return 'X'
        if num == PLAYER_O:
            return 'O'
        return '_'

    print(f"\n    | A | B | C |")

    i = 0
    for y in range(3):
        print(f"| {Y_AXIS_LABELS[y]} |", end='')
        for _ in range(3):
            print(f" {convert(_grid[i])} |", end='')
            i += 1


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
    play = player_turn(_grid)
    actions_list = [(play, i) for i in range(len(_grid)) if _grid[i] == FIELD_EMPTY]
    return actions_list


def result(_grid, action):
    (play, index) = action
    grid_copy = _grid.copy()
    grid_copy[index] = play
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



