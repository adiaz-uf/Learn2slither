"""
UI module.

Provides Pygame-based components for displaying the game board and navigating
between lobby, game-over, and statistics screens.
"""

import pygame
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'AI_Model'))
from Board import Board  # noqa: E402

# Color palette (R, G, B)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 0, 0)
GREEN = (0, 200, 0)
YELLOW = (200, 200, 0)
GRAY = (50, 50, 50)
DARK_GREEN = (0, 150, 0)
BLUE = (0, 100, 200)


class StartScreen:
    """Displays a start screen with a board-size selector (10 / 14 / 18)."""

    BOARD_SIZES = [10, 14, 18]

    def __init__(self, screen_width=800, screen_height=600,
                 allowed_sizes=None):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.display = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption('Learn to Slither - Start Game')

        self.font_large = pygame.font.Font(None, 72)
        self.font_medium = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)

        sizes = allowed_sizes if allowed_sizes else self.BOARD_SIZES
        self.allowed_sizes = sizes

        button_width = 180
        button_height = 80
        spacing = 30
        total_width = len(sizes) * button_width + (len(sizes) - 1) * spacing
        start_x = (screen_width - total_width) // 2
        button_y = screen_height // 2 + 30

        self.size_buttons = []
        for i, size in enumerate(sizes):
            btn = Button(
                start_x + i * (button_width + spacing),
                button_y,
                button_width, button_height,
                f"{size} x {size}", DARK_GREEN, GREEN,
            )
            self.size_buttons.append((size, btn))

    def run(self):
        """Event loop. Returns ('start', board_size) or ('quit', None)."""
        while True:
            mouse_pos = pygame.mouse.get_pos()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return 'quit', None
                if event.type == pygame.MOUSEBUTTONDOWN:
                    for size, btn in self.size_buttons:
                        if btn.is_clicked(event.pos):
                            return 'start', size

            for _, btn in self.size_buttons:
                btn.check_hover(mouse_pos)

            self.display.fill(BLACK)

            title = self.font_large.render("Learn to Slither", True, GREEN)
            self.display.blit(title, title.get_rect(
                center=(self.screen_width // 2, 150)))

            subtitle = self.font_medium.render(
                "Choose board size", True, WHITE)
            self.display.blit(subtitle, subtitle.get_rect(
                center=(self.screen_width // 2, 250)))

            if len(self.allowed_sizes) < len(self.BOARD_SIZES):
                hint = self.font_small.render(
                    "(this model only supports the sizes shown)",
                    True, YELLOW,
                )
                self.display.blit(hint, hint.get_rect(
                    center=(self.screen_width // 2, 295)))

            for _, btn in self.size_buttons:
                btn.draw(self.display, self.font_medium)
            pygame.display.flip()


class Button:
    """A simple clickable button with a hover effect."""

    def __init__(self, x, y, width, height, text, color, hover_color,
                 text_color=WHITE):
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

    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)


class GameOverScreen:
    """Displays the game-over score and offers 'New Game' or 'View Stats'."""

    def __init__(self, final_score, screen_width=800, screen_height=600):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.display = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption('Learn to Slither - Game Over')

        self.final_score = final_score
        self.font_large = pygame.font.Font(None, 72)
        self.font_medium = pygame.font.Font(None, 48)

        button_width = 250
        button_height = 80
        button_y = screen_height // 2 + 50

        self.start_new_game_btn = Button(
            screen_width // 2 - button_width - 20,
            button_y, button_width, button_height,
            "New Game", DARK_GREEN, GREEN,
        )
        self.view_stats_btn = Button(
            screen_width // 2 + 20,
            button_y, button_width, button_height,
            "View Stats", BLUE, (0, 150, 255),
        )

    def run(self):
        """Event loop. Returns 'new_game', 'stats', or 'quit'."""
        while True:
            mouse_pos = pygame.mouse.get_pos()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return 'quit'
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if self.start_new_game_btn.is_clicked(event.pos):
                        return 'new_game'
                    if self.view_stats_btn.is_clicked(event.pos):
                        return 'stats'

            self.start_new_game_btn.check_hover(mouse_pos)
            self.view_stats_btn.check_hover(mouse_pos)

            self.display.fill(BLACK)

            title = self.font_large.render("Game Over", True, RED)
            self.display.blit(title, title.get_rect(
                center=(self.screen_width // 2, 200)))

            score_text = self.font_medium.render(
                f"Final Score: {self.final_score}", True, WHITE)
            self.display.blit(score_text, score_text.get_rect(
                center=(self.screen_width // 2, 280)))

            self.start_new_game_btn.draw(self.display, self.font_medium)
            self.view_stats_btn.draw(self.display, self.font_medium)
            pygame.display.flip()


class StatsScreen:
    """Displays cumulative session statistics and a back button."""

    def __init__(self, screen_width=800, screen_height=600, stats=None):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.display = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption('Learn to Slither - Statistics')

        self.stats = stats if stats else {
            'total_games': 0,
            'total_score': 0,
            'high_score': 0,
            'average_score': 0.0,
        }
        self.font_large = pygame.font.Font(None, 72)
        self.font_medium = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 36)

        button_width = 200
        button_height = 60
        self.back_btn = Button(
            screen_width // 2 - button_width // 2,
            screen_height - 100,
            button_width, button_height,
            "Back", DARK_GREEN, GREEN,
        )

    def run(self):
        """Event loop. Returns 'back' or 'quit'."""
        while True:
            mouse_pos = pygame.mouse.get_pos()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return 'quit'
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if self.back_btn.is_clicked(event.pos):
                        return 'back'

            self.back_btn.check_hover(mouse_pos)
            self.display.fill(BLACK)

            title = self.font_large.render("Statistics", True, GREEN)
            self.display.blit(title, title.get_rect(
                center=(self.screen_width // 2, 100)))

            stats_rows = [
                ("Total Games", self.stats['total_games']),
                ("Total Score", self.stats['total_score']),
                ("High Score", self.stats['high_score']),
                ("Average Score", f"{self.stats['average_score']:.1f}"),
            ]
            y = 200
            for label, value in stats_rows:
                line = self.font_medium.render(
                    f"{label}: {value}", True, WHITE)
                self.display.blit(line, line.get_rect(
                    center=(self.screen_width // 2, y)))
                y += 60

            self.back_btn.draw(self.display, self.font_medium)
            pygame.display.flip()


class BoardGameUI:
    """Pygame window that renders the game board driven by Board logic."""

    BLOCK_SIZE = 80  # Pixel size of each board cell

    def __init__(self, board_size=12):
        """
        Args:
            board_size: full board size including wall border
                        (playable area = board_size - 2).
        """
        pygame.init()
        self.board_size = board_size
        self.w = board_size * self.BLOCK_SIZE
        self.h = board_size * self.BLOCK_SIZE

        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Learn to Slither')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)

        self.board = Board(board_size)
        self.board.initialize()
        self.game_over = False

    def _draw_cell(self, row, col, color):
        """Draw a single cell at grid position (row, col)."""
        pygame.draw.rect(
            self.display, color,
            pygame.Rect(col * self.BLOCK_SIZE, row * self.BLOCK_SIZE,
                        self.BLOCK_SIZE, self.BLOCK_SIZE),
        )
        pygame.draw.rect(
            self.display, GRAY,
            pygame.Rect(col * self.BLOCK_SIZE, row * self.BLOCK_SIZE,
                        self.BLOCK_SIZE, self.BLOCK_SIZE),
            1,
        )

    def _update_ui(self):
        """Render the current board state to the pygame window."""
        self.display.fill(BLACK)

        cell_colors = {
            'W': GRAY, 'H': BLUE, 'S': YELLOW,
            'G': GREEN, 'R': RED, '0': BLACK,
        }
        for i in range(self.board_size):
            for j in range(self.board_size):
                cell = self.board.board[i][j]
                self._draw_cell(i, j, cell_colors.get(cell, BLACK))

        # Score = snake head + body length
        score = len(self.board.snake.body) + 1
        score_text = self.font.render(f'Score: {score}', True, WHITE)
        self.display.blit(score_text, (10, 10))

        if self.game_over:
            msg = self.font.render('GAME OVER!', True, RED)
            msg_rect = msg.get_rect(center=(self.w // 2, self.h // 2))
            pygame.draw.rect(self.display, BLACK, msg_rect.inflate(20, 20))
            self.display.blit(msg, msg_rect)

        pygame.display.flip()
