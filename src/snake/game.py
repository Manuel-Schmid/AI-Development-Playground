import pygame
import numpy as np
from enum import Enum
from collections import namedtuple
import random

pygame.init()


class Direction(Enum):
    UP = 1
    DOWN = 2
    RIGHT = 3
    LEFT = 4


Point = namedtuple('Point', 'x, y')

BLOCK_SIZE = 20
SPEED = 40


class SnakeGame:
    w = 640
    h = 480
    display = pygame.display.set_mode((w, h))
    pygame.display.set_caption('Snake')
    clock = pygame.time.Clock()
    direction = Direction.RIGHT

    head = Point(w / 2, h / 2)
    snake = [head, Point(head.x - BLOCK_SIZE, head.y), Point(head.x - (2 * BLOCK_SIZE), head.y)]

    score = 0
    food = None
    frame_iteration = 0

    def __init__(self):
        self.reset()

    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        head = Point(self.w / 2, self.h / 2)
        self.snake = [head,
                      Point(head.x - BLOCK_SIZE, head.y),
                      Point(head.x - (2 * BLOCK_SIZE), head.y)]

        score = 0
        self.food = None
        self.place_food()
        self.frame_iteration = 0

    def place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self.place_food()

    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move
        self._move(action)  # update the head
        self.snake.insert(0, self.head)

        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self.place_food()
        else:
            self.snake.pop()

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score

    def has_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False

    def update_gui(self):
        self.display.fill((0, 0, 0))

        for pt in self.snake:
            pygame.draw.rect(self.display, (0, 0, 255), pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, (0, 100, 255), pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, (200, 0, 0), pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        # text = font.render("Score: " + str(self.score), True, (255, 255, 255))
        # self.display.blit(text, [0, 0])
        pygame.display.flip()

