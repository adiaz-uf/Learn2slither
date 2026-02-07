import numpy as np

class Snake:
    def __init__(self):
        self.head = []
        self.body = []

    def init_snake(self, head_position: list, body_position1: list, body_position2: list):
        self.head = head_position

        body = body_position1
        self.body.append(body)

        body = body_position2
        self.body.append(body)

    def grow(self, new_body_position: list):
        self.body.append(new_body_position)

    def move(self):
        pass    

