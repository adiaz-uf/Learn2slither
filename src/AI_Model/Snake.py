"""
Snake module.

Defines the Snake data structure: head position and body segment list.
All movement logic is handled by Board.
"""


class Snake:
    """Represents the snake with a head and a list of body segments."""

    def __init__(self):
        self.head = []
        self.body = []

    def init_snake(
        self,
        head_position: list,
        body_position1: list,
        body_position2: list
    ):
        """Initialize the snake with three contiguous segments."""
        self.head = head_position
        self.body.append(body_position1)
        self.body.append(body_position2)

    def grow(self, new_body_position: list):
        """Append a new segment to the tail of the snake."""
        self.body.append(new_body_position)
