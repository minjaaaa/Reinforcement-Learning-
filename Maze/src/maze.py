import numpy as np
import matplotlib.pyplot as plt
from random import Random 
from threading import Event
from itertools import chain, repeat
import copy
from board import MazeBoard, symbols
from cells import Cell, TelCell, Position, Actions, RegCell
from typing import Optional, Callable
random = Random()
running = Event()

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
    
    def next_state(self, s:Position, a:Actions) -> tuple[Optional[Position], Optional[float]]:
        #return the next state and reward after taking action a in state s
        if a == Actions.UP.name:
            next_s = Position(s.row - 1, s.col)
        elif a == Actions.DOWN.name:
            next_s = Position(s.row + 1, s.col)
        elif a == Actions.LEFT.name:
            next_s = Position(s.row, s.col - 1)
        elif a == Actions.RIGHT.name:
            next_s = Position(s.row, s.col + 1)
        else:
            return None, None
        
        #teleport state handling, if agent is on a teleport cell, it is actually teleported
        if self.board[s()].is_teleport():
            cell = self.board[next_s()]
            if isinstance(cell, TelCell):
                next_s = cell.get_next_cell()

        #if next state is valid and steppable, return it
        if 0 <= next_s.row < self.board.rows_no and \
            0 <= next_s.col < self.board.cols_no and \
            self.board[next_s()].is_steppable():

            if self.board[next_s()].is_teleport():
                cell = self.board[next_s()]
                if isinstance(cell, TelCell):
                    next_s = cell.get_next_cell()
        
            return next_s, self.board[next_s()].get_reward()
        return None, None
                


    def update_state_value(self, s:Position, v:dict[Position,float], gamma:float, actions) -> float:
        #return max value - iterating over all possible actions
        values=[]
        for a in actions:
            next_state, reward = self.next_state(s, a)
            if next_state is not None and reward is not None:
                values.append(reward + gamma * v[next_state])
        #if no actions were possible return small value
        return  max(values) if len(values) else v[s]-10
    
    def update_all_state_values(self, v:dict[Position,float], gamma:float, policy:dict) -> dict[Position,float]:
        #return updated state values for all states by following the given policy
        values = copy.deepcopy(v)
        for s in self.states:
            if not self.board[s()].is_terminal():
                actions = [action for action in self.actions if action in policy[s]]

                if self.board[s()].is_teleport():
                    cell = self.board[s()]
                    if isinstance(cell, TelCell):
                        next_state = cell.get_next_cell()
                        actions = [action for action in self.actions if action in policy[next_state]]
                values[s] = self.update_state_value(s, v, gamma, actions)
        return values
    
    def update_action_value(self, s:Position, a:Actions, q:dict[Position, dict[Actions, float]], gamma:float, policy:dict)->float:
        #return updated Q-value for state s and action a 
        q_values = []
        next_state, reward = self.next_state(s, a)
        if next_state is None:
            return 0
        elif self.board[next_state()].is_terminal():
            return 0
        
        available_actions = [act for act in Actions if act in policy.get(next_state, {})]
        for next_a in available_actions:
            q_val = q[next_state].get(next_a)
            if q_val is None or reward is None:
                continue
            q_values.append(reward + gamma * q_val)
        #if no actions were possible return small value or None
        return max(q_values) if q_values else 0.0
    
    def update_all_action_values(self, q:dict[Position, dict[Actions,float]], gamma:float, policy:dict)->dict[Position,dict[Actions,float]]:
    #return updated Q-values for all states and actions by following the given policy
        q_values = copy.deepcopy(q)
        for s in self.states:
            if not self.board[s()].is_terminal():
                for a in Actions:
                    q_values[s][a] = self.update_action_value(s, a, q, gamma, policy)
        return q_values
    
         


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

def value_iteration(update:Callable, values: dict, gamma:float, eps:float, iterations:int = 100, policy:Optional[dict] = None) ->dict[Position, float]:
    #perform value iteration for a given number of iterations or until convergence
    new_values = copy.deepcopy(values)
    old_values = copy.deepcopy(values)
    error = float('inf')
    
    for i in range(iterations):
        new_values = update(old_values, gamma, policy if policy is not None else {})
        error = get_error(copy.deepcopy(new_values), copy.deepcopy(old_values))
        old_values = copy.deepcopy(new_values)
        if error < eps:
            print(f"Value iteration converged after {i} iterations.")
            return new_values
    print(f"Value iteration did not converge after {iterations} iterations. Final error: {error}")
    return new_values

def policy_iteration(env: MazeEnvironment, update:Callable, values:dict, policy:dict, update_policy:Callable, gamma:float, eps:float, iterations:int = 100) -> tuple[dict, dict[Position,float]]:
    #perform policy iteration for a given number of iterations or until convergence
    new_values = copy.deepcopy(values)
    
    for i in range(iterations):
        old_values = copy.deepcopy(values)
        new_values= value_iteration(update, old_values, gamma, eps, iterations, policy=policy)
        new_policy = {s:update_policy(env, s, new_values, gamma).name for s in env.states}
        if policy == new_policy:
            print(f"Policy iteration converged after {i} iterations.")
            return policy, new_values
        policy = new_policy
    #policy did not converge
    print(f"Policy iteration did not converge after {iterations} iterations.")
    return policy, new_values

def greedy(env:MazeEnvironment, s:Position, v:dict[Position,float], gamma:float) -> Actions:
    #return the action that maximizes the value function in state s
    values=[]
    min_v=min(v.values())

    for a in Actions:
        s_new, r = env.next_state(s,a)

        if s_new is not None and r is not None:
            values.append(r + gamma * v[s_new])
        else:
            values.append(min_v - 1000)  #penalize impossible actions
    return Actions(np.argmax(values))

def greedy_q(env:MazeEnvironment, s:Position, q:dict[Position, dict[Actions,float]], gamma:float) -> Actions:
    #return the action that maximizes the Q-value function in state s
    values=[(q[s][a],a) for a in Actions if q[s][a] is not None]

    if not len(values):
        actions=list(env.actions.keys())
        return random.choice(list(Actions))
    max_value = max(values, key=lambda x: x[0])
    return max_value[1]

def apply_policy(env:MazeEnvironment, policy:Callable, state:Position, gamma:float, values:dict[Position,float], pi:Optional[dict]=None) -> float:
    #apply the given policy starting from start_state for max_steps
    #simulates an apisode and returns the total reward
    gain=0
    i=0
    while not env.board[state()].is_terminal() and running.is_set():
        if pi is not None:
            action = pi[state]
        else:
            action = policy(env, state, values, gamma)
        
        next_state, reward = env.next_state(state, action)
        if next_state is None or reward is None:
            break
        state = next_state
        gain+=(gamma**i)*reward
        i+=1
        symbol=symbols[env.actions[action.name]]

        env.board.ax.set_title(f"Gamma={gamma}, Gain={gain:.1f}")
        env.board.draw_agent(state(), symbol)
    return gain