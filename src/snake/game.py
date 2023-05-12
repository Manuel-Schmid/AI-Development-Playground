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

        head = Point(w / 2, h / 2)
        self.snake = [head,
                      Point(head.x - BLOCK_SIZE, head.y),
                      Point(head.x - (2 * BLOCK_SIZE), head.y)]

        score = 0
        self.food = None
        self.place_food()
        self.frame_iteration = 0

    def place_food(self):
        x = random.randint(0, (w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self.place_food()

