from dataclasses import dataclass, FrozenInstanceError
from enum import Enum

@dataclass(frozen=True)
class Position():
    row : int
    col : int

    def __call__(self):
        return (self.row, self.col)

# To add more actions edit Actions, symbols and next_state method of Environment
class Actions(Enum):
    UP       = 0
    DOWN     = 1
    LEFT     = 2
    RIGHT    = 3

class Cell():
    
    def __init__(self, reward, steppable, terminal, teleport):
        self.reward = reward
        self.steppable = steppable
        self.terminal = terminal
        self.teleport = teleport

    def set_reward(self, reward):
        self.reward = reward
    
    def get_reward(self) -> float:
        return self.reward

    def is_steppable(self) -> bool:
        return self.steppable

    def is_terminal(self) -> bool:
        return self.terminal

    def is_teleport(self) -> bool:
        return self.teleport

class RegCell(Cell):

    def __init__(self, reward):
        super().__init__(reward=reward, 
                        steppable=True, 
                        terminal=False,
                        teleport=False)

class TermCell(Cell):

    def __init__(self, reward):
        super().__init__(reward=reward, 
                        steppable=True, 
                        terminal=True,
                        teleport=False)

class WallCell(Cell):

    def __init__(self):
        super().__init__(reward=0, 
                        steppable=False, 
                        terminal=False,
                        teleport=False)

class TelCell(Cell):

    def __init__(self, next_cell : Position):
        super().__init__(reward=0, 
                        steppable=True, 
                        terminal=False,
                        teleport=True)

        self.next_cell = next_cell
    
    def get_next_cell(self):
        return self.next_cell
