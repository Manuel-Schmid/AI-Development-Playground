import numpy as np
from collections import deque
from game import Direction, Point
from model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.has_collision(point_r)) or
            (dir_l and game.has_collision(point_l)) or
            (dir_u and game.has_collision(point_u)) or
            (dir_d and game.has_collision(point_d)),

            # Danger right
            (dir_u and game.has_collision(point_r)) or
            (dir_d and game.has_collision(point_l)) or
            (dir_l and game.has_collision(point_u)) or
            (dir_r and game.has_collision(point_d)),

            # Danger left
            (dir_d and game.has_collision(point_r)) or
            (dir_u and game.has_collision(point_l)) or
            (dir_r and game.has_collision(point_u)) or
            (dir_l and game.has_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]

        return np.array(state, dtype=int)

