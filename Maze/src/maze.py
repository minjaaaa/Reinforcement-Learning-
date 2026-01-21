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
import time

class MazeEnvironment():

    def __init__(self, board:MazeBoard, ):
        self.board = board
        self.states = self.init_states()
        self.actions = self.init_actions()
    
    def init_states(self)->list[Position]:
        #return all steppable positions on the board
        #process cells for teleports
        states = []
        print(f"Board size: {self.board.rows_no} x {self.board.cols_no}")
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
    
    def next_state(self, s:Position, a:str) -> tuple[Optional[Position], Optional[float]]:
        #return the next state and reward after taking action a in state s
        if self.board[s()].is_teleport():
            cell = self.board[s()]
            if isinstance(cell, TelCell):
                teleport_dest = cell.get_next_cell()
        
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
        
        if not (0 <= next_s.row < self.board.rows_no and
            0 <= next_s.col < self.board.cols_no):
        # Out of bounds - stay at current position with penalty
            return s, -1

        # Check if intended next position is steppable
        if not self.board[next_s()].is_steppable():
            return s, -1  # Wall or non-steppable - invalid move
        
        # Handle teleport: if next_s is a teleport, get the actual destination
        if self.board[next_s()].is_teleport():
            cell = self.board[next_s()]
            if isinstance(cell, TelCell):
                teleport_dest = cell.get_next_cell()
                
                # Validate teleport destination
                if teleport_dest is None:
                    return None, None
                
                if not (0 <= teleport_dest.row < self.board.rows_no and
                        0 <= teleport_dest.col < self.board.cols_no):
                    # Teleport destination is out of bounds - invalid
                    print(f"Warning: Teleport at ({next_s.row},{next_s.col}) leads to out-of-bounds ({teleport_dest.row},{teleport_dest.col})")
                    return None, None
                
                if not self.board[teleport_dest()].is_steppable():
                    # Teleport destination is not steppable - invalid
                    return None, None
                
                # Use teleport destination as the actual next state
                next_s = teleport_dest
    
        # Return the final next state and its reward
        return next_s, self.board[next_s()].get_reward()
                


    def update_state_value(self, s:Position, v:dict[Position,float], gamma:float, actions: list[str]) -> float:
        #return max value - iterating over all possible actions
        cell = self.board[s()]
        if self.board[s()].is_teleport():
            cell = self.board[s()]
            if isinstance(cell, TelCell):
                next_state = cell.get_next_cell()
                return cell.get_reward() + gamma * v[next_state]
        values=[]
        for a in actions:
            next_state, reward = self.next_state(s, a)
            if next_state is not None and reward is not None:
                if next_state in v:
                    values.append(reward + gamma * v[next_state])
                
        #if no actions were possible return small value
        return  max(values) if len(values) else v[s]-10
    
    def update_all_state_values(self, v:dict[Position,float], gamma:float, policy:dict) -> dict[Position,float]:
        #return updated state values for all states by following the given policy
        values = copy.deepcopy(v)
        for s in self.states:
            if self.board[s()].is_terminal():
                continue
            #if not self.board[s()].is_terminal():
            #    actions = [action for action in self.actions if action in policy.get(s, [])]

            #    if self.board[s()].is_teleport():
            #        cell = self.board[s()]
            #        if isinstance(cell, TelCell):
            #            next_state = cell.get_next_cell()
            #            if not self.board[next_state()].is_terminal():
            #                actions = [action for action in self.actions if action in policy[next_state]]
            #    values[s] = self.update_state_value(s, v, gamma, actions)
            actions = [a for a in self.actions if a in policy.get(s, [])]
            values[s] = self.update_state_value(s, v, gamma, actions)
        return values
    
    def update_action_value(self, s:Position, a:str, q:dict[Position, dict[str, float]], gamma:float, policy:Optional[dict] = None)->float:
        #return updated Q-value for state s and action a 
        q_values: list[float] = []
        next_state, reward = self.next_state(s, a)
        if next_state is None:
            return 0
        elif self.board[next_state()].is_terminal():
            return 0
        
        allowed = []
        if policy is None or len(policy) == 0:  # ← Dodaj proveru
            # if no policy given - all actions are allowed
            allowed_actions = set(self.actions.keys())
        else:
            # have policy - only actions allowed by it
            allowed = policy.get(next_state, [])
        
        if isinstance(allowed, dict):
            allowed_actions = set(allowed.keys())
        elif isinstance(allowed, list):
            allowed_actions = set(allowed)
        elif isinstance(allowed, str):
            allowed_actions = {allowed}
        else:
            # Default: SVE akcije
            allowed_actions = set(self.actions.keys())

        # Računaj Q-vrednost
        for action_name in allowed_actions:
            q_val = q[next_state].get(action_name)
            if q_val is not None and reward is not None:
                q_values.append(reward + gamma * q_val)
        
        return max(q_values) if q_values else 0.0
    
    def update_all_action_values(self, q:dict[Position, dict[str,float]], gamma:float, policy:dict)->dict[Position,dict[str,float]]:
    #return updated Q-values for all states and actions by following the given policy
        q_values = copy.deepcopy(q)
        for s in self.states:
            if not self.board[s()].is_terminal():
                for a in self.actions:
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
        old_values = new_values
        if error < eps:
            print(f"Value iteration converged after {i} iterations.")
            return new_values
    print(f"Value iteration did not converge after {iterations} iterations. Final error: {error}")
    return new_values

def policy_iteration(env: MazeEnvironment, update:Callable, values:dict, policy:Optional[dict], update_policy:Callable, gamma:float, eps:float, iterations:int = 100) -> tuple[dict, dict[Position,float]]:
    #perform policy iteration for a given number of iterations or until convergence
    if policy is None:
        policy = {}
    new_values = copy.deepcopy(values)
    
    for i in range(iterations):
        old_values = copy.deepcopy(new_values)
        new_values= value_iteration(update, old_values, gamma, eps, iterations, policy=policy)
        new_policy = {s:update_policy(env, s, new_values, gamma).name for s in env.states}
        if policy == new_policy and policy is not None:
            print(f"Policy iteration converged after {i} iterations.")
            return policy, new_values
        policy = new_policy
    #policy did not converge
    print(f"Policy iteration did not converge after {iterations} iterations.")
    return policy, new_values if policy is not None else {}

def greedy(env:MazeEnvironment, s:Position, v:dict[Position,float], gamma:float) -> Actions:
    #return the action that maximizes the value function in state s
    values={}
    min_v=min(v.values())
    best_action_name:str =""
    
    for a in env.actions:
        s_new, r = env.next_state(s,a)

        if s_new is not None and r is not None:
            values[a]=(r + gamma * v[s_new])
        else:
            values[a]=(min_v - 1000)  #penalize impossible actions
    #find best action
    best_action_name = max(values, key=lambda x: values[x])
    
    
    return Actions[best_action_name]

def greedy_q(env:MazeEnvironment, s:Position, q:dict[Position, dict[str,float]], gamma:float) -> Actions:
    #return the action that maximizes the Q-value function in state s
    if s not in q:
        # Random action if no Q-values
        return Actions[random.choice(list(env.actions.keys()))]
    action_values = [(q[s][action_name], action_name) 
                     for action_name in env.actions 
                     if q[s].get(action_name) is not None]
    if not action_values:
        # Random action if no valid Q-values
        return Actions[random.choice(list(env.actions.keys()))]

    max_q_value, best_action_name = max(action_values, key=lambda x: x[0])
    return Actions[best_action_name]

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
        # Convert to string if needed
        if isinstance(action, Actions):
            action_name = action.name  # Actions.UP → 'UP'
        else:
            action_name = action
        
        next_state, reward = env.next_state(state, action_name)

        if next_state is None or reward is None:
            break

        state = next_state
        gain+=(gamma**i)*reward
        i+=1
        
        # update display
        symbol = symbols[env.actions[action_name]]
        env.board.ax.set_title(f"Gamma={gamma}, Gain={gain:.1f}, Step={i}")
        env.board.draw_agent(state(), symbol)
        
        
        plt.pause(0.5) 
        env.board.fig.canvas.draw()
        env.board.fig.canvas.flush_events()
    return gain

def in_range(s:str, rows_n:int, cols_n:int) -> bool:
    #check if the given string represents a valid position on the board
    try:
        numbers = s.split(",")
        r = int(numbers[0])
        c = int(numbers[1])

        if 0<=r<rows_n and 0<=c<cols_n:
            return True
    except Exception as e:
        print("Wrong format")
    
    return False

def get_start_position(cells:list[list[Cell]], rows_no:int, cols_no:int) -> tuple[int,int]:
    #Keep prompting the user for a starting position until they enter:
    #valid form, position within bounds, and steppable cell.
    prompts = chain(["Input starting position (r,c): "], \
              repeat(f"Rows must be in range (0 - {rows_no}) \nColumns must be in range (0 - {cols_no}) \nChosen cell must be steppable \nTry again: "))
    replies = map(input, prompts)

    valid = next(filter(lambda s: in_range(s, rows_no, cols_no) and cells[int(s[0])][int(s[2])].is_steppable(), replies))
    
    return int(valid[0]), int(valid[2])

def get_iteration_method() -> int:
    '''Validates method chosen by the user'''

    prompts = chain(["1. Value iteration\n2. Policy iteration\nChoose a iteration method: "], \
                    repeat(f"Acceptable answers are '1' or '2'.\nTry again: "))
    replies = map(input, prompts)

    valid = next(filter(lambda s: int(s) in [1, 2], replies)) #Check if result is in list [1, 2]
    return int(valid)

def get_simulation_values() -> int:
    '''Validates method chosen by the user'''

    prompts = chain(["1. State-values\n2. Q-values\nChoose which values are used: "], \
                    repeat(f"Acceptable answers are '1' or '2'.\nTry again: "))
    replies = map(input, prompts)

    valid = next(filter(lambda s: s.isnumeric() and int(s) in [1, 2], replies)) #if replie is number digit and is in list [1, 2]
    return int(valid)

def get_board() -> MazeBoard:
    '''Generates a board with dimensions chosen by the user'''

    prompts = chain(["Input board dimensions (r,c): "], repeat("Row and column numbers have to positive integers \nTry again: "))
    replies = map(input, prompts)

    valid = next(filter(lambda s: in_range(s, 100, 100), replies))
    valid = valid.split(",")
    r, c = int(valid[0]), int(valid[1])

    # Allows the user to generate boards until they are satisfied
    while True:
        board = MazeBoard(r, c)
        board.ax.set_title("Board preview")
        board.draw_board()

        if continue_question("Do you want to continue? (y/n): "):
            return board
        
        #plt.close()
        plt.close(board.fig)
def draw_policy(env, policy, ax=None, title="Optimal Policy"):
    """
    Vizualizuj policy sa strelicama na svakoj ćeliji.
    
    Args:
        env: MazeEnvironment objekat
        policy: dict {Position: action_name} ili {Position: Actions}
        ax: matplotlib axes (opcionalno, kreira novi ako nije prosleđen)
        title: Naslov plota
    """

    
    # Kreiraj axes ako nije prosleđen
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.figure
    
    board = env.board
    
    # Nacrtaj tablu (pozadina sa bojama)
    board_img = np.ones((board.rows_no, board.cols_no, 3), dtype=np.uint8)
    teleport_index = 0

    
    # Prikaži sliku table
    ax.imshow(board_img, 
             extent=(0, board.cols_no, board.rows_no, 0),
             origin='upper',
             interpolation='nearest',
             aspect='equal',
             zorder=1)
    
    # Mapa akcija na simbole strelica
    action_symbols = {
        'UP': '↑',
        'DOWN': '↓',
        'LEFT': '←',
        'RIGHT': '→'
    }
    
    # Nacrtaj policy strelice
    for s, action in policy.items():
        # Konvertuj action u string ako je Actions enum
        if isinstance(action, Actions):
            action_name = action.name
        else:
            action_name = action
        
        # Dobavi simbol strelice
        symbol = action_symbols.get(action_name, '?')
        
        # Nacrtaj strelicu u centru ćelije
        ax.text(s.col + 0.5, s.row + 0.5,
               symbol,
               fontsize=20,
               fontweight='bold',
               color='black',
               ha='center',
               va='center',
               zorder=10)
    
    # Podesi grid
    ax.set_xticks(np.arange(0, board.cols_no + 1, 1), minor=True)
    ax.set_yticks(np.arange(0, board.rows_no + 1, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=1, zorder=5)
    
    # Postavi major ticks
    ax.set_xticks(range(board.cols_no))
    ax.set_yticks(range(board.rows_no))
    
    # Labele osa
    ax.set_xlabel('Column', fontsize=12)
    ax.set_ylabel('Row', fontsize=12)
    
    # Naslov
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Postavi limite
    ax.set_xlim(0, board.cols_no)
    ax.set_ylim(board.rows_no, 0)
    
    # Ažuriraj prikaz
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    return fig, ax

def continue_question(msg:str) -> bool:
    prompts = chain([msg], \
                    repeat(f"Only 'y' or 'n' are valid responses. \n{msg}"))
    replies = map(input, prompts)

    valid = next(filter(lambda s: s == 'y' or s == 'n', replies))

    print("------------------------------")

    if valid == 'y':
        return True
    return False


# Event handler za tastaturu
def on_key(event):
    if event.key == 'q':
        print("\nExiting Q-value exploration...")
        exploring[0] = False
        plt.close(board.fig)


if __name__ == "__main__":
    running.set()

    plt.ion()
    board = get_board()

    env = MazeEnvironment(board)

    gamma = float(input("Input gamma: "))
    eps = float(input("Input eps tolerance: "))

    # Initializes state and state-action values randomly
    v = {s:env.board[s()].get_reward() for s in env.states}
    q = {}
    for s in env.states:
        if not board[s()].is_terminal():
            q[s] = {}
            for a in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                sn, _ = env.next_state(s,a)
                if sn is not None:
                    q[s][a] = (-10)*random.random()
                else:
                    q[s][a] = None
        else:
            q[s] = {a: None for a in env.actions}

    mode = get_iteration_method()
    actions = list(env.actions.keys())
    #print(f"Calculating values...:{actions}")

    values: dict[Position, float] = v
    q_values = q

    policy_v: Optional[dict] = None
    policy_q: Optional[dict] = None

    gain = float(0)

    if mode == 1:
        # When value iteration is used all actions are possible
        policy = {s: actions for s in env.states if not board[s()].is_terminal()}
    
        values = value_iteration(env.update_all_state_values, v, gamma, eps, policy=policy)
        q_values = value_iteration(env.update_all_action_values, q, gamma, eps, policy=policy)
    elif mode == 2:
        # When policy iteraton is used only actions defined by the policy are possible
        policy_v_init = {s: random.choice(list(env.actions.keys())) for s in env.states if not board[s()].is_terminal()}

        print("Calculating state values with policy iteration...")
        policy_v, values = policy_iteration(env, env.update_all_state_values, v, policy_v_init, greedy,gamma,eps)
        print("Calculating Q-values with policy iteration...")
        policy_q, q_values = policy_iteration(env,env.update_all_action_values,q,None, greedy_q, gamma, eps)
        #draw_policy(env, policy_v, title="Optimal Policy - Value Iteration")
    print("#################################################")
    print("\n" + "="*60)
    print("DIAGNOSTIC CHECK")
    print("="*60)

    # 1. Proveri mode
    print(f"Mode: {mode}")

    # 2. Proveri da li su vrednosti izračunate
    if 'values' in locals():
        print(f"✓ Values exist: {len(values)} states")
    else:
        print("✗ Values DO NOT exist")

    # 3. Proveri policy
    if 'policy_v' in locals():
        print(f"Policy_v type: {type(policy_v)}")
        
        if policy_v is None:
            print("✗ ERROR: policy_v is None!")
        elif not policy_v:
            print("✗ WARNING: policy_v is empty dict")
        else:
            print(f"✓ policy_v exists: {len(policy_v)} states")
            
            # Prikaži sample
            sample_states = list(policy_v.items())[:3]
            for s, a in sample_states:
                print(f"  {s} → {a}")
    else:
        print("✗ ERROR: policy_v variable does not exist!")

    print("="*60 + "\n")

    # Sada pozovi draw_policy SAMO ako policy postoji
    if 'policy_v' in locals() and policy_v is not None and policy_v:
        draw_policy(env, policy_v)
    else:
        print("Skipping draw_policy - policy not available")
    print("#################################################")
    # Clear and prepare board
    board.ax.clear()
    plt.close('all')

    max_w, max_h = 8, 6
    aspect = board.cols_no / board.rows_no

    if aspect >= max_w / max_h:
        fig_width = max_w
        fig_height = max_w / aspect
    else:
        fig_height = max_h
        fig_width = max_h * aspect

    board.fig, board.ax = plt.subplots(figsize=(fig_width, fig_height))
    plt.ion()
    board.draw_board()
    board.ax.set_title("Board values preview")

    #print("Drawing values...")
    board.draw_values(values)
    #print(len(board.ax.texts))

    #print(board.fig)
    #print(plt.get_fignums())   # lista aktivnih figura
    #print(board.ax)
    plt.show(block=False)
    plt.pause(0.1)


    # Show the figure explicitly
    try:
        #board.fig.show()
        print("✓ Figure.show() called")
    except Exception as e:
        print(f"Figure.show() failed: {e}")

    print("Click on a cell to view it's action values.")
    print("Press 'q' to continue.")
    print("="*60 + "\n")

    # Flag za kontrolu
    exploring = [True]

    def on_click_wrapper(event):
        board.on_mouse_click(event, q_values)

    click_cid = board.fig.canvas.mpl_connect('button_press_event', on_click_wrapper)
    key_cid = board.fig.canvas.mpl_connect('key_press_event', on_key)


    while 1:
        res = plt.waitforbuttonpress(8)
        if res is None:
            if continue_question("Do you want to continue? (y/n): "):
                continue
            break
        
        if board.mouse_col is not None and board.mouse_row is not None:
            board.draw_q_values(q_values)
            plt.pause(0.01)
        if not exploring[0]:
            break
    board.fig.canvas.mpl_disconnect(click_cid)
    board.fig.canvas.mpl_disconnect(key_cid)

    print("Continuing to simulation...")
    
    for text in board.value_texts:
        text.remove()
    board.value_texts.clear()

    for text in board.action_texts:
        text.remove()
    board.action_texts.clear()

    for text in board.moves:
        text.remove()
    board.moves.clear()
    plt.pause(0.5)

    try:
        #print("usao u try")
        while running.is_set():
            #print("usao u while")
            val = get_simulation_values()
            #draw_policy(env, policy_v if mode == 1 else policy_q, ax=board.ax, title="Optimal Policy")
            # Resets the board
            for text in board.moves:
                text.remove()
            board.moves.clear()
            plt.pause(0.5)

            r, c = get_start_position(board.cells, board.rows_no, board.cols_no)
            s = Position(r,c)

            board.ax.clear()
            board.draw_board()
            board.ax.set_title(f"Gamma={gamma}, Gain={0}")
            board.draw_agent(s(), "+")
            plt.pause(0.5)

            # State values
            if val == 1:
                gain = apply_policy(env, greedy, s, gamma, values, policy_v)
            # State-action values
            elif val == 2:
                gain = apply_policy(env, greedy_q, s, gamma, q_values, policy_q)

            # Shows all moves made in this run
            for text in board.moves:
                text.set_visible(True)
            plt.pause(0.5)

            print("Gain: " + str(gain)) 

            if not continue_question("Do you want to play again? (y/n): "):
                running.clear()

    except KeyboardInterrupt:
        running.clear()

    plt.close('all')
    print("END OF GAME")