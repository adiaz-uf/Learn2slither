import pygame
import random
import numpy as np
import sys
import os

# Add path to import Board
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'AI_Model'))
from Agent import Agent
from Board import Board

# Colors (R, G, B)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 0, 0)
GREEN = (0, 200, 0)
YELLOW = (200, 200, 0)
GRAY = (50, 50, 50)
DARK_GREEN = (0, 150, 0)
LIGHT_GRAY = (100, 100, 100)
BLUE = (0, 100, 200)

class Button:
    def __init__(self, x, y, width, height, text, color, hover_color, text_color=WHITE):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.is_hovered = False
        
    def draw(self, screen, font):
        color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, WHITE, self.rect, 2)
        
        text_surface = font.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)
    
    def check_hover(self, pos):
        self.is_hovered = self.rect.collidepoint(pos)
        return self.is_hovered
    
    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)

class LobbyScreen:
    def __init__(self, screen_width=800, screen_height=600):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.display = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption('Learn to Slither - Lobby')
        
        self.font_large = pygame.font.Font(None, 72)
        self.font_medium = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 36)
        
        # Board size options
        self.board_sizes = [(10, 10), (14, 14), (18, 18)]
        self.selected_size = None
        
        # Create buttons for board sizes
        button_width = 200
        button_height = 80
        button_spacing = 30
        start_y = screen_height // 2 - 50
        
        self.size_buttons = []
        for i, (w, h) in enumerate(self.board_sizes):
            x = screen_width // 2 - button_width // 2
            y = start_y + i * (button_height + button_spacing)
            text = f"{w}x{h}"
            btn = Button(x, y, button_width, button_height, text, DARK_GREEN, GREEN)
            self.size_buttons.append(btn)
    
    def run(self):
        running = True
        while running:
            mouse_pos = pygame.mouse.get_pos()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    # Use event.pos for accurate click position
                    for i, button in enumerate(self.size_buttons):
                        if button.is_clicked(event.pos):
                            self.selected_size = self.board_sizes[i]
                            return self.selected_size
            
            # Update hover states
            for button in self.size_buttons:
                button.check_hover(mouse_pos)
            
            # Draw
            self.display.fill(BLACK)
            
            # Title
            title = self.font_large.render("Learn to Slither", True, GREEN)
            title_rect = title.get_rect(center=(self.screen_width // 2, 150))
            self.display.blit(title, title_rect)
            
            # Subtitle
            subtitle = self.font_medium.render("Select Board Size", True, WHITE)
            subtitle_rect = subtitle.get_rect(center=(self.screen_width // 2, 220))
            self.display.blit(subtitle, subtitle_rect)
            
            # Draw buttons
            for button in self.size_buttons:
                button.draw(self.display, self.font_medium)
            
            pygame.display.flip()
        
        return None

class GameOverScreen:
    def __init__(self, final_score, screen_width=800, screen_height=600):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.display = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption('Learn to Slither - Game Over')
        
        self.final_score = final_score
        self.font_large = pygame.font.Font(None, 72)
        self.font_medium = pygame.font.Font(None, 48)
        
        # Create buttons
        button_width = 250
        button_height = 80
        button_y = screen_height // 2 + 50
        
        self.start_new_game_btn = Button(
            screen_width // 2 - button_width - 20,
            button_y,
            button_width,
            button_height,
            "Start New Game",
            DARK_GREEN,
            GREEN
        )
        
        self.view_stats_btn = Button(
            screen_width // 2 + 20,
            button_y,
            button_width,
            button_height,
            "View Stats",
            BLUE,
            (0, 150, 255)
        )
    
    def run(self):
        running = True
        while running:
            mouse_pos = pygame.mouse.get_pos()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return 'quit'
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    # Use event.pos for accurate click position
                    if self.start_new_game_btn.is_clicked(event.pos):
                        return 'new_game'
                    elif self.view_stats_btn.is_clicked(event.pos):
                        return 'stats'
            
            # Update hover states
            self.start_new_game_btn.check_hover(mouse_pos)
            self.view_stats_btn.check_hover(mouse_pos)
            
            # Draw
            self.display.fill(BLACK)
            
            # Game Over text
            game_over_text = self.font_large.render("Game Over", True, RED)
            game_over_rect = game_over_text.get_rect(center=(self.screen_width // 2, 200))
            self.display.blit(game_over_text, game_over_rect)
            
            # Final score
            score_text = self.font_medium.render(f"Final Score: {self.final_score}", True, WHITE)
            score_rect = score_text.get_rect(center=(self.screen_width // 2, 280))
            self.display.blit(score_text, score_rect)
            
            # Draw buttons
            self.start_new_game_btn.draw(self.display, self.font_medium)
            self.view_stats_btn.draw(self.display, self.font_medium)
            
            pygame.display.flip()
        
        return 'quit'

class StatsScreen:
    def __init__(self, screen_width=800, screen_height=600, stats=None):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.display = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption('Learn to Slither - Statistics')
        
        self.stats = stats if stats else {
            'total_games': 0,
            'total_score': 0,
            'high_score': 0,
            'average_score': 0.0
        }
        
        self.font_large = pygame.font.Font(None, 72)
        self.font_medium = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 36)
        
        # Back button
        button_width = 200
        button_height = 60
        self.back_btn = Button(
            screen_width // 2 - button_width // 2,
            screen_height - 100,
            button_width,
            button_height,
            "Back",
            DARK_GREEN,
            GREEN
        )
    
    def run(self):
        running = True
        while running:
            mouse_pos = pygame.mouse.get_pos()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return 'quit'
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    # Use event.pos for accurate click position
                    if self.back_btn.is_clicked(event.pos):
                        return 'back'
            
            # Update hover state
            self.back_btn.check_hover(mouse_pos)
            
            # Draw
            self.display.fill(BLACK)
            
            # Title
            title = self.font_large.render("Statistics", True, GREEN)
            title_rect = title.get_rect(center=(self.screen_width // 2, 100))
            self.display.blit(title, title_rect)
            
            # Stats
            y_offset = 200
            line_spacing = 60
            
            stats_data = [
                ("Total Games", self.stats['total_games']),
                ("Total Score", self.stats['total_score']),
                ("High Score", self.stats['high_score']),
                ("Average Score", f"{self.stats['average_score']:.1f}")
            ]
            
            for label, value in stats_data:
                text = f"{label}: {value}"
                stat_text = self.font_medium.render(text, True, WHITE)
                stat_rect = stat_text.get_rect(center=(self.screen_width // 2, y_offset))
                self.display.blit(stat_text, stat_rect)
                y_offset += line_spacing
            
            # Draw back button
            self.back_btn.draw(self.display, self.font_medium)
            
            pygame.display.flip()
        
        return 'quit'

class BoardGameUI:
    """Game UI integrated with Board.py logic"""
    def __init__(self, board_size=10):
        pygame.init()
        
        self.block_size = 80  # Size of each cell
        self.board_size = board_size
        self.w = (board_size) * self.block_size
        self.h = (board_size) * self.block_size
        
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Learn to Slither')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        
        # Initialize board logic
        self.board = Board(board_size)
        self.board.initialize()
        
        self.game_over = False
        self.score = 0
    
    def play_step(self, direction):
        """Execute one game step and return reward, game_over, score"""
        if self.game_over:
            return 0, True, self.score
        
        # Get current snake length
        old_length = len(self.board.snake.body)

        self.board.get_snake_view()
        result, reward = self.board.move_snake(direction)
        print(reward)
        
        if result is None:
            self.game_over = True
            return -10, True, self.score
        
        # Update score based on length change
        new_length = len(self.board.snake.body)
        reward = 0
        
        if new_length > old_length:
            self.score += 10  # Green apple
            reward = 10
        elif new_length < old_length:
            self.score -= 5   # Red apple
            reward = -5
        
        return reward, False, self.score
    
    def draw_cell(self, x, y, color):
        """Draw a single cell on the board"""
        pygame.draw.rect(
            self.display, 
            color, 
            pygame.Rect(y * self.block_size, x * self.block_size, self.block_size, self.block_size)
        )
        # Draw border
        pygame.draw.rect(
            self.display, 
            GRAY, 
            pygame.Rect(y * self.block_size, x * self.block_size, self.block_size, self.block_size),
            1
        )
    
    def _update_ui(self):
        """Render the game board"""
        self.display.fill(BLACK)
        
        # Draw the board based on the state
        for i in range(self.board_size):
            for j in range(self.board_size):
                cell = self.board.board[i][j]
                
                if cell == 'W':  # Wall
                    self.draw_cell(i, j, GRAY)
                elif cell == 'H':  # Head
                    self.draw_cell(i, j, BLUE)
                elif cell == 'S':  # Snake body
                    self.draw_cell(i, j, YELLOW)
                elif cell == 'G':  # Green apple
                    self.draw_cell(i, j, GREEN)
                elif cell == 'R':  # Red apple
                    self.draw_cell(i, j, RED)
                elif cell == '0':  # Empty
                    self.draw_cell(i, j, BLACK)
        
        # Draw score
        score_text = self.font.render(f'Score: {self.score}', True, WHITE)
        self.display.blit(score_text, (10, 10))
        
        # Draw game over message
        if self.game_over:
            game_over_text = self.font.render('GAME OVER!', True, RED)
            text_rect = game_over_text.get_rect(center=(self.w // 2, self.h // 2))
            # Draw background for text
            pygame.draw.rect(self.display, BLACK, text_rect.inflate(20, 20))
            self.display.blit(game_over_text, text_rect)
        
        pygame.display.flip()

class GameManager:
    """
    Simple game manager for UI screens (lobby, game over, stats).
    Training logic has been moved to main.py
    """
    def __init__(self, board_size=10):
        pygame.init()
        self.stats = {
            'total_games': 0,
            'total_score': 0,
            'high_score': 0,
            'average_score': 0.0
        }
        self.current_board_size = board_size
    
    def update_stats(self, score):
        """Update statistics after a game"""
        self.stats['total_games'] += 1
        self.stats['total_score'] += score
        if score > self.stats['high_score']:
            self.stats['high_score'] = score
        self.stats['average_score'] = self.stats['total_score'] / self.stats['total_games']
    
    def run(self):
        """Main game loop with lobby and screens"""
        while True:
            # Show lobby to select board size
            lobby = LobbyScreen()
            board_size = lobby.run()
            
            if board_size is None:
                break  # User closed the window
            
            # Set the board size (only width matters, it's square)
            self.current_board_size = board_size[0]
            
            print(f"\nSelected board size: {self.current_board_size}x{self.current_board_size}")
            print("To start training, run main.py with your desired configuration")
            print("This UI is for demonstration purposes only.\n")
            
            # For now, just show game over screen
            # Training should be done via main.py
            game_over_screen = GameOverScreen(self.stats['high_score'])
            choice = game_over_screen.run()
            
            if choice == 'new_game':
                continue  # Go back to lobby
            elif choice == 'stats':
                # Show stats screen
                stats_screen = StatsScreen(stats=self.stats)
                stats_choice = stats_screen.run()
                if stats_choice == 'back':
                    # Show game over screen again
                    game_over_screen = GameOverScreen(self.stats['high_score'])
                    choice = game_over_screen.run()
                    if choice == 'new_game':
                        continue
                    elif choice == 'quit':
                        break
                elif stats_choice == 'quit':
                    break
            elif choice == 'quit':
                break
        
        pygame.quit()

if __name__ == "__main__":
    print("Note: For training, please run main.py instead.")
    print("This file is for UI components only.\n")
    manager = GameManager()
    manager.run()
    pygame.quit()
