BOARD_EMPTY = 0
BOARD_PLAYER_X = 1
BOARD_PLAYER_O = -1

X_AXIS_LABELS = ["A", "B", "C"]
Y_AXIS_LABELS = ["1", "2", "3"]


def print_board(s):
    def convert(num):
        if num == BOARD_PLAYER_X:
            return 'X'
        if num == BOARD_PLAYER_O:
            return 'O'
        return '_'

    print(f"\n    | A | B | C |")

    i = 0
    for y in range(3):
        print(f"| {Y_AXIS_LABELS[y]} |", end='')
        for _ in range(3):
            print(f" {convert(s[i])} |", end='')
            i += 1


