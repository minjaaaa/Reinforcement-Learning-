import numpy as np
import matplotlib.pyplot as plt
from random import Random 
from threading import Event
from itertools import chain, repeat
import copy
from board import MazeBoard, symbols
from cells import *

class MazeEnvironment():

    def __init__(self, board:MazeBoard):
        self.board = board
        self.states = self.init_states()
        self.actions = self.init_actions()
    
    def init_states(self)->list[Position]:
        #return all steppable positions on the board
        #process cells for teleports
        states = []
        for r in range(self.board.rows_no):
            for c in range(self.board.cols_no):
                position = Position(r, c)
                if not self.board[position()].is_steppable():
                    continue

                states.append(position)
                cell = self.board[position()]
                if cell.is_teleport():
                    if isinstance(cell, TelCell):
                        next_cell = cell.get_next_cell()
                        # The reward for the teleport cell is the same as for the 
                        # cell it leads to, if it's possible to teleport to it
                        if self.board[next_cell()].is_steppable() and position != next_cell:
                            self.board[position()].set_reward(self.board[next_cell()].get_reward())
                        else:
                            # Teleports that lead into walls are treated as regular cells
                            self.board.cells[position.row][position.col] = RegCell(-1)
                            print(f"Teleport on ({position.row}, {position.col}) was removed.")
        return states
    
    def init_actions(self) -> dict[str, int]:
        #return a dict of all possible actions
        actions={}
        for a in Actions:
            actions[a.name] = a.value
        
        return actions
    


#OUT OF CLASS
def get_error(new:dict, old:dict) -> float:
    #returns the maximum error between old and new value states
    key = next(iter(new)) 

    #in case of state value dictionaries
    if isinstance(new[key], (int, float)) or new[key] is None:
        new = {k: (0 if v is None else v) for k, v in new.items()}
        old = {k: (0 if v is None else v) for k, v in old.items()}
        err = max([abs(new[x] - old[x]) for x in new])
        return err
    
    #in case of Q-value dictionaries
    max_val = []
    for s in new:
        max_val.append(get_error(new[s], old[s])) #create a list of max errors for each state
    return max(max_val)