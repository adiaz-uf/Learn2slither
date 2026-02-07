from time import sleep
import numpy as np

from Snake import Snake

# REWARD VALUES
INSTANT_GAMEOVER = -10  # Reduced from -100 to prevent loss explosion
GREEN_APPLE = 10       # Good reward for eating
RED_APPLE = -5         # Not used anymore (no red apples)
NO_EAT = -0.01         # Very small penalty to encourage exploration

class Board ():
    def __init__(self, boardSize: int):
        self.boardSize = boardSize
        self.board = np.zeros((boardSize, boardSize), dtype=str)
        self.snake = Snake()
        self.food = []

    def initialize(self):
        for i in range(self.boardSize):
            for j in range(self.boardSize):
                if i == 0 or i == self.boardSize-1 or j == 0 or j == self.boardSize-1:
                    self.board[i][j] = 'W'
                else:
                    self.board[i][j] = '0'

        # First, generate a random head position
        head_x = np.random.randint(1, self.boardSize - 1)
        head_y = np.random.randint(1, self.boardSize - 1)
        head_position = [head_x, head_y]
        self.board[head_position[0]][head_position[1]] = 'H'
        
        # Then get adjacent positions for the body
        body_pos_1 = self.get_adyacent_position(head_position)
        self.board[body_pos_1[0]][body_pos_1[1]] = 'S'
        body_pos_2 = self.get_adyacent_position(body_pos_1)
        self.board[body_pos_2[0]][body_pos_2[1]] = 'S'
        
        # Finally initialize the snake with all positions
        self.snake.init_snake(head_position, body_pos_1, body_pos_2)
        
        self.init_food()


    def init_food(self):
        # Add green apple
        while len(self.food) < 2:
            x = np.random.randint(1, self.boardSize - 1)
            y = np.random.randint(1, self.boardSize - 1)
            position = [x, y]
            if position not in self.snake.body and position != self.snake.head:
                self.food.append((x, y, 'G'))  # Green apple
                self.board[x][y] = 'G'
        
        # Add red apple
        while len(self.food) < 3:
            x = np.random.randint(1, self.boardSize - 1)
            y = np.random.randint(1, self.boardSize - 1)
            position = [x, y]
            if position not in self.snake.body and position != self.snake.head:
                self.food.append((x, y, 'R'))  # Red apple
                self.board[x][y] = 'R'
            
    def check_collision(self, position_to_move: list):
        if position_to_move[0] <= 0 or position_to_move[0] >= self.boardSize - 1\
        or position_to_move[1] <= 0 or position_to_move[1] >= self.boardSize - 1:
            return True
        return False
    
    def check_eat(self, position_to_move: list):
        if self.board[position_to_move[0], position_to_move[1]] == 'G':
            return 'G'
        elif self.board[position_to_move[0], position_to_move[1]] == 'R':
            return 'R'
        else:
            return None
        

    def add_apple(self, position: list, type: str):
        # Remove the eaten apple from food list
        self.food = [apple for apple in self.food if not (apple[0] == position[0] and apple[1] == position[1])]
        
        # Add new apple of the same type
        while True:
            x = np.random.randint(1, self.boardSize - 1)
            y = np.random.randint(1, self.boardSize - 1)
            new_position = [x, y]
            
            # Check if position is free
            if new_position not in self.snake.body and new_position != self.snake.head:
                # Check if no other food is at this position
                position_free = True
                for apple in self.food:
                    if apple[0] == x and apple[1] == y:
                        position_free = False
                        break
                
                if position_free:
                    self.food.append((x, y, type))
                    self.board[x][y] = type
                    return
                
    def get_snake_view(self):
        snake_head_x = self.snake.head[0]
        snake_head_y = self.snake.head[1]

        view = [[], [], [], []]  # [UP, DOWN, LEFT, RIGHT] - matches action indices!

        # UP view (cells above head) - index 0
        for cell in range(0, snake_head_x):
            view[0].append(self.board[cell][snake_head_y])
        
        # DOWN view (cells below head) - index 1
        for cell in range(snake_head_x + 1, self.boardSize):
            view[1].append(self.board[cell][snake_head_y])
        
        # LEFT view (cells to the left of head) - index 2
        for cell in range(0, snake_head_y):
            view[2].append(self.board[snake_head_x][cell])
        
        # RIGHT view (cells to the right of head) - index 3
        for cell in range(snake_head_y + 1, self.boardSize):
            view[3].append(self.board[snake_head_x][cell])
        
        # Debug: Print in cross format (optional, only if needed)
        # Uncomment these lines to visualize the snake's view
        # print(' ' * len(view[2]) + ''.join(view[0]))  # UP
        # print(''.join(view[2]) + 'H' + ''.join(view[3]))  # LEFT + HEAD + RIGHT
        # print(' ' * len(view[2]) + ''.join(view[1]))  # DOWN
        # print()  # Empty line
        
        return view

    def get_adyacent_position(self, position: list):
        position_x = position[0]
        position_y = position[1]

        directions = [
            [position_x - 1, position_y],  # UP
            [position_x + 1, position_y],  # DOWN
            [position_x, position_y - 1],  # LEFT
            [position_x, position_y + 1]   # RIGHT
        ]
        
        # Shuffle directions
        np.random.shuffle(directions)
        
        for new_pos in directions:
            if self.board[new_pos[0]][new_pos[1]] == '0':
                return new_pos
        
        return None
    
    def move_snake(self, direction: str):
        print(direction, '\n')
        # Calculate new head position based on direction
        old_head_x = self.snake.head[0]
        old_head_y = self.snake.head[1]
        
        new_x = old_head_x
        new_y = old_head_y

        if direction == 'UP':
            new_x = old_head_x - 1
        elif direction == 'DOWN':
            new_x = old_head_x + 1
        elif direction == 'LEFT':
            new_y = old_head_y - 1
        elif direction == 'RIGHT':
            new_y = old_head_y + 1
        
        new_snake_head = [new_x, new_y]

        # Check collision with walls
        if self.check_collision(new_snake_head):
            return None, INSTANT_GAMEOVER
        
        # Check collision with snake body
        if new_snake_head in self.snake.body:
            return None, INSTANT_GAMEOVER
        
        # Check if eating food
        food_type = self.check_eat(new_snake_head)
        
        # Clear old head position on board
        self.board[old_head_x][old_head_y] = 'S'
        
        # Move body: each segment moves to the position of the previous one
        if food_type == 'G':
            # Green apple: grow (don't remove tail)
            self.snake.body.insert(0, [old_head_x, old_head_y])
            self.add_apple(new_snake_head, 'G')
        elif food_type == 'R':
            # Red apple: shrink (remove one segment)
            # Game over if body is already empty (only head left)
            if len(self.snake.body) == 0:
                return None, INSTANT_GAMEOVER  # Game over - can't shrink anymore
            
            # Remove tail (last segment)
            tail = self.snake.body.pop()
            self.board[tail[0]][tail[1]] = '0'
            
            # Move remaining body (if any)
            self.snake.body.insert(0, [old_head_x, old_head_y])
            self.add_apple(new_snake_head, 'R')
        else:
            # Normal move: body follows head, tail is removed
            self.snake.body.insert(0, [old_head_x, old_head_y])
            tail = self.snake.body.pop()
            self.board[tail[0]][tail[1]] = '0'
        
        # Update head position
        self.snake.head = [new_x, new_y]
        self.board[new_x][new_y] = 'H'

        if food_type == 'G':
            reward = GREEN_APPLE
        elif food_type == 'R':
            reward = RED_APPLE
        else:
            reward = NO_EAT
        
        return new_snake_head, reward
    
    def get_distance_to_closest_food(self):
        """Calculate Manhattan distance to closest green apple"""
        if not self.food:
            return 0
        
        head_x, head_y = self.snake.head
        min_dist = float('inf')
        
        for food_x, food_y, food_type in self.food:
            if food_type == 'G':  # Only consider green apples
                dist = abs(head_x - food_x) + abs(head_y - food_y)
                min_dist = min(min_dist, dist)
        
        return min_dist if min_dist != float('inf') else 0
        
    def print_colored_board(self):
        COLORS = {
            'R': '\033[91m',  # RED
            'G': '\033[92m',  # GREEN
            'H': '\033[94m',  # BLUE
            'S': '\033[93m',  # YELLOW
            'W': '\033[90m',  # GRAY
            '0': '\033[0m',   # RESET
        }
        RESET = '\033[0m'
        
        for row in self.board:
            for cell in row:
                color = COLORS.get(cell, '\033[0m')
                print(f"{color}{cell}{RESET}", end=' ')
            print()
        print()


    def reset(self):
        self.board = np.zeros((self.boardSize, self.boardSize), dtype=int)

    def get_state(self):
        return self.board