import pygame
import numpy as np
from enum import Enum
from collections import namedtuple

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


