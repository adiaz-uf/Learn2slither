"""
Board module.

Manages the game grid, snake placement, food spawning, movement logic,
collision detection, and reward calculation.
"""

import numpy as np

from Snake import Snake

# Reward constants.
INSTANT_GAMEOVER = -100
GREEN_APPLE = 50
RED_APPLE = -10
NO_EAT = -0.3


class Board:
    """
    Represents the game board.

    The board is a square NumPy array of strings where:
      'W' = wall, '0' = empty, 'H' = snake head,
      'S' = snake body, 'G' = green apple, 'R' = red apple.

    boardSize includes the wall border, so the playable area is
    (boardSize - 2) x (boardSize - 2).
    """

    def __init__(self, board_size: int):
        self.boardSize = board_size
        self.board = np.zeros((board_size, board_size), dtype=str)
        self.snake = Snake()
        self.food = []

    def initialize(self):
        """Fill the board with walls, spawn the snake, and place food."""
        # Draw walls on the border and clear the interior
        for i in range(self.boardSize):
            for j in range(self.boardSize):
                if (i == 0 or i == self.boardSize - 1
                        or j == 0 or j == self.boardSize - 1):
                    self.board[i][j] = 'W'
                else:
                    self.board[i][j] = '0'

        # Place snake head at a random interior position
        head_x = np.random.randint(1, self.boardSize - 1)
        head_y = np.random.randint(1, self.boardSize - 1)
        head_position = [head_x, head_y]
        self.board[head_position[0]][head_position[1]] = 'H'

        # Place two adjacent body segments
        body_pos_1 = self.get_adjacent_position(head_position)
        self.board[body_pos_1[0]][body_pos_1[1]] = 'S'
        body_pos_2 = self.get_adjacent_position(body_pos_1)
        self.board[body_pos_2[0]][body_pos_2[1]] = 'S'

        self.snake.init_snake(head_position, body_pos_1, body_pos_2)
        self.init_food()

    def init_food(self):
        """Spawn 2 green apples and 1 red apple at free positions."""
        # Place 2 green apples
        while len(self.food) < 2:
            x = np.random.randint(1, self.boardSize - 1)
            y = np.random.randint(1, self.boardSize - 1)
            occupied = {(f[0], f[1]) for f in self.food}
            if ([x, y] not in self.snake.body
                    and [x, y] != self.snake.head
                    and (x, y) not in occupied):
                self.food.append((x, y, 'G'))
                self.board[x][y] = 'G'

        # Place 1 red apple
        while len(self.food) < 3:
            x = np.random.randint(1, self.boardSize - 1)
            y = np.random.randint(1, self.boardSize - 1)
            occupied = {(f[0], f[1]) for f in self.food}
            if ([x, y] not in self.snake.body
                    and [x, y] != self.snake.head
                    and (x, y) not in occupied):
                self.food.append((x, y, 'R'))
                self.board[x][y] = 'R'

    def check_collision(self, position: list) -> bool:
        """Return True if the position is on or outside the wall border."""
        return (position[0] <= 0 or position[0] >= self.boardSize - 1
                or position[1] <= 0 or position[1] >= self.boardSize - 1)

    def check_eat(self, position: list):
        """Return 'G', 'R', or None depending on what is at the position."""
        cell = self.board[position[0], position[1]]
        if cell == 'G':
            return 'G'
        if cell == 'R':
            return 'R'
        return None

    def add_apple(self, eaten_position: list, apple_type: str):
        """Remove the eaten apple and spawn a new one of the same type."""
        self.food = [
            apple for apple in self.food
            if not (apple[0] == eaten_position[0]
                    and apple[1] == eaten_position[1])
        ]

        while True:
            x = np.random.randint(1, self.boardSize - 1)
            y = np.random.randint(1, self.boardSize - 1)
            occupied = {(f[0], f[1]) for f in self.food}
            if ([x, y] not in self.snake.body
                    and [x, y] != self.snake.head
                    and [x, y] != eaten_position
                    and (x, y) not in occupied):
                self.food.append((x, y, apple_type))
                self.board[x][y] = apple_type
                return

    def get_snake_view(self) -> list:
        """
        Return the snake's cross-shaped vision as four lists:
        [UP, DOWN, LEFT, RIGHT], each ordered from closest to farthest cell.
        Indices match agent action indices (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT).
        """
        hx = self.snake.head[0]
        hy = self.snake.head[1]
        view = [[], [], [], []]

        # UP: rows above the head (closest first)
        for cell in range(hx - 1, -1, -1):
            view[0].append(self.board[cell][hy])

        # DOWN: rows below the head (closest first)
        for cell in range(hx + 1, self.boardSize):
            view[1].append(self.board[cell][hy])

        # LEFT: columns to the left of the head (closest first)
        for cell in range(hy - 1, -1, -1):
            view[2].append(self.board[hx][cell])

        # RIGHT: columns to the right of the head (closest first)
        for cell in range(hy + 1, self.boardSize):
            view[3].append(self.board[hx][cell])

        return view

    def get_adjacent_position(self, position: list):
        """
        Return a random free ('0') cell adjacent to the given position.
        Returns None if all neighbours are occupied.
        """
        px, py = position[0], position[1]
        directions = [
            [px - 1, py],
            [px + 1, py],
            [px, py - 1],
            [px, py + 1],
        ]
        np.random.shuffle(directions)
        for new_pos in directions:
            if self.board[new_pos[0]][new_pos[1]] == '0':
                return new_pos
        return None

    def move_snake(self, direction: str):
        """
        Move the snake one step in the given direction.

        Returns:
            (new_head_position, reward) on a valid move, or
            (None, INSTANT_GAMEOVER) on collision — wall, self, or
            zero-length after red apple.
        """
        old_hx = self.snake.head[0]
        old_hy = self.snake.head[1]

        if direction == 'UP':
            new_hx, new_hy = old_hx - 1, old_hy
        elif direction == 'DOWN':
            new_hx, new_hy = old_hx + 1, old_hy
        elif direction == 'LEFT':
            new_hx, new_hy = old_hx, old_hy - 1
        else:  # RIGHT
            new_hx, new_hy = old_hx, old_hy + 1

        new_head = [new_hx, new_hy]

        # Collision with wall
        if self.check_collision(new_head):
            return None, INSTANT_GAMEOVER

        # Collision with own body
        if new_head in self.snake.body:
            return None, INSTANT_GAMEOVER

        food_type = self.check_eat(new_head)

        # Mark the old head cell as body
        self.board[old_hx][old_hy] = 'S'

        if food_type == 'G':
            # Grow: prepend old head to body, do not remove tail
            self.snake.body.insert(0, [old_hx, old_hy])
            self.add_apple(new_head, 'G')

        elif food_type == 'R':
            # Shrink: game over if the body is already empty
            if len(self.snake.body) == 0:
                return None, INSTANT_GAMEOVER
            # Remove the tail before inserting new body segment
            tail = self.snake.body.pop()
            self.board[tail[0]][tail[1]] = '0'
            self.snake.body.insert(0, [old_hx, old_hy])
            self.add_apple(new_head, 'R')

        else:
            # Normal move: body follows head, tail is removed
            self.snake.body.insert(0, [old_hx, old_hy])
            tail = self.snake.body.pop()
            self.board[tail[0]][tail[1]] = '0'

        # Update head position on the board
        self.snake.head = new_head
        self.board[new_hx][new_hy] = 'H'

        if food_type == 'G':
            return new_head, GREEN_APPLE
        if food_type == 'R':
            return new_head, RED_APPLE
        return new_head, NO_EAT
