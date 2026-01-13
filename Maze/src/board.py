import numpy as np
import math
import matplotlib.pyplot as plt
from cells import *
from random import Random
from cells import *

random = Random()

symbols = {
    0 : '↑',
    1 : '↓',
    2 : '←',
    3 : '→'
}


class MazeBoard():

    def __init__(self, rows_no, cols_no):
        cells = self.generate_cells(rows_no, cols_no)
        rows, cols, cells = self.process_cells(cells)
        self.rows_no = rows
        self.cols_no = cols
        self.cells = cells
        self.mouse_row: int =0
        self.mouse_col: int =0
        
        # Saves all moves made by the agent in the explotation faze for visualization purposes
        self.moves = []

        self.value_texts = []
        self.action_texts = []

        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(left=0, right=self.rows_no)
        self.ax.set_ylim(top=self.cols_no, bottom=0)
        self.ax.invert_yaxis()
        self.ax.set_xticks(np.arange(0, rows, step=1))
        self.ax.set_yticks(np.arange(cols-1, -1, step=-1))

        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        
    
    def __getitem__(self, key: tuple[int, int]) -> Cell:
        r, c = key
        return self.cells[r][c]

    def process_cells(self, cells: list[list[Cell]]):
        cells = [list(row) for row in cells]

        if not cells:
            raise Exception("Number of rows in a board must be at least one")
        if not cells[0]:
            raise Exception("Number of colums in a board must be at least one")
        
        rows_no = len(cells)
        cols_no = len(cells[0])
        for row in cells:
            if not row or len(row) != cols_no:
                raise Exception("All rows in a board must have the same number of colums")
        return rows_no, cols_no, cells
    
    def onclick(self, event):
        """Handle mouse click events."""
        # Only process clicks on the main axis
        if event.inaxes != self.ax:
            return
        
        # Get clicked position
        if event.xdata is not None and event.ydata is not None:
            col = int(event.xdata)
            row = int(event.ydata)
            
            
            # Check bounds
            if 0 <= row < self.rows_no and 0 <= col < self.cols_no:
                self.mouse_row = row
                self.mouse_col = col
                
            # Draw Q-values
            print(f"Showing Q-values for cell ({row}, {col})")
            board.draw_q_values(q_values)
    
    def int_to_cell(self, code:int, rows_no:int, cols_no:int) -> Cell:
        #Converts an integer code to a Cell object
        #0 - regular cell, 1 - penalty cell, 2 - wall cell, 3 - teleport cell, 4 - terminal cell
        if code == 0:
            return RegCell(reward=-1.0)
        elif code == 1:
            return RegCell(reward=-10.0)
        elif code == 2:
            return WallCell()
        elif code == 3:
            return TelCell(Position(
                random.randint(0, rows_no-1), 
                random.randint(0, cols_no-1)
                ))
        elif code == 4:
            return TermCell(reward=0.0)
        else:
            raise ValueError(f"Invalid cell code: {code}. Expected 0-4.")
        
    def generate_cells(self, rows_no, cols_no) -> list[list[Cell]]:
        # Distribution of regular, penalty, wall and teleport cells
        # Cells distribution example:   ENUM
        # RegularCell (rw=-1) 10/15,     0
        # RegularCell (rw=-10) 2/15,     1
        # WallCell 2/15,                 2
        # TeleportCell 1/15]             3
        
        # Creates a 2D array of integers representing cell types
        cell_codes = np.random.choice(4, size=(rows_no, cols_no), p=[0.67, 0.13, 0.13, 0.07])
        # random.choice goes from 0 to 3
        # A terminal cell is placed in a random place on the board, overwriting whatever cell was there
        cell_codes[random.randint(0, rows_no-1), random.randint(0, cols_no-1)] = 4


        cells = [[self.int_to_cell(cell_codes[i, j], rows_no, cols_no) for j in range(cols_no)] for i in range(rows_no)]
        return cells
    
    def draw_board(self):
        board_img=np.ones((self.rows_no, self.cols_no, 3), dtype=np.uint8)
        teleport_index = 0
        for i in range(self.rows_no):
            for j in range(self.cols_no):
                cell = self[i, j]
                if isinstance(cell, RegCell):
                    if cell.get_reward() == -1:
                        board_img[i, j, :] = [255, 255, 255] # Regular cell, WHITE COLOR
                    else:
                        board_img[i, j, :] = [255, 0, 0] # Regular cell with penalty, RED COLOR
                elif isinstance(cell, WallCell):
                    board_img[i, j, :] = [0, 0, 0] # Wall cell, BLACK COLOR
                elif isinstance(cell, TelCell):
                    board_img[i, j, :] = [0, 0, 255] # Teleport cell, BLUE COLOR  

                    self.ax.text(j+0.5, i+0.5, str(teleport_index), color="pink", fontweight="bold", fontsize=10,ha="center",va="center",zorder=5)
                    #print(f"Teleport {teleport_index}: {i}, {j}")
                    next_cell = cell.get_next_cell()
                    self.ax.text(next_cell.col+0.5, next_cell.row+0.5, str(teleport_index),color="pink",  fontweight="bold", fontsize=10)
                    #print(f"Teleport to {teleport_index}: {next_cell.row}, {next_cell.col}")

                    teleport_index += 1  
                else:
                    board_img[i, j, :] = [255, 255, 0] # Terminal cell, YELLOW
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.imshow(board_img, extent=(0, self.cols_no, self.rows_no, 0), origin="upper", interpolation='nearest',zorder=0)
        

    def hide_text(self, text):
        text.set_visible(False)
    
    def show_text(self, text):
        text.set_visible(True)

    def draw_values(self, values: dict[Position, float]):
        """Draw state values on the board."""
        
        # Clear previous value texts
        for text in self.value_texts:
            text.remove()  # Actually remove from plot
        self.value_texts.clear()  # Clear the list
        
        self.ax.set_xlim(-0.5, self.cols_no - 0.5)
        self.ax.set_ylim(-0.5, self.rows_no - 0.5)
        self.ax.invert_yaxis()
        
        # Draw new values
        drawn_count=0
        for k, v in values.items():
            if v is not None:
                if 0 <= k.row < self.rows_no and 0 <= k.col < self.cols_no:
                    #print(k.row, k.col, v)
                    #print(self.ax.get_xlim(), self.ax.get_ylim())
                    text = self.ax.text(k.col+0.5, k.row+0.5, 
                                    f"{values[k]:.1f}",  # No need for str() wrapper
                                    fontsize=10, color='black',
                                    ha='center', va='center', zorder=5)
                    self.value_texts.append(text)
                    drawn_count +=1
        
        # Update display
        self.ax.figure.canvas.draw()
        self.ax.figure.canvas.flush_events()
    
    def draw_q_values(self, q_values):
        """Draw Q-values for the cell under the mouse."""
        
        # Clear previous texts
        for text in self.value_texts:
            text.remove()
        self.value_texts.clear()
        
        for text in self.action_texts:
            text.remove()
        self.action_texts.clear()
        
        # Get mouse position (default to 0,0 if out of bounds)
        row = self.mouse_row if self.mouse_row is not None else 0
        col = self.mouse_col if self.mouse_col is not None else 0
        s = Position(row, col)
        
        try:
            # Check if position is valid
            if not (0 <= row < self.rows_no and 0 <= col < self.cols_no):
                print(f"Position ({row}, {col}) is out of bounds")
                return
            
            cell = self[s.row, s.col]

            # Check if it's a wall
            if isinstance(cell, WallCell):
                print(f"Position ({row}, {col}) is a WALL - no Q-values")
                # Optionally show a message on the board
                text = self.ax.text(s.col + 0.5, s.row + 0.5,
                                "WALL",
                                fontsize=12, color='white',
                                ha='center', va='center',
                                bbox=dict(boxstyle='round', 
                                        facecolor='black', 
                                        alpha=0.8))
                self.action_texts.append(text)
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                return
            # Check if it's a terminal state
            if cell.is_terminal():
                print(f"Position ({row}, {col}) is TERMINAL")
                # Show terminal message
                text = self.ax.text(s.col + 0.5, s.row + 0.5,
                                "GOAL",
                                fontsize=14, color='green',
                                ha='center', va='center',
                                weight='bold',
                                bbox=dict(boxstyle='round', 
                                        facecolor='yellow', 
                                        alpha=0.8))
                self.action_texts.append(text)
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                return
            # Check if position has Q-values
            if s not in q_values:
                print(f"Position ({row}, {col}) has NO Q-VALUES in dictionary")
                print(f"Cell type: {type(cell).__name__}")
                print(f"Is steppable: {cell.is_steppable()}")
                print(f"Is terminal: {cell.is_terminal()}")
                
                # Show message on board
                text = self.ax.text(s.col + 0.5, s.row + 0.5,
                                "No Q-values",
                                fontsize=10, color='red',
                                ha='center', va='center',
                                bbox=dict(boxstyle='round', 
                                        facecolor='yellow', 
                                        alpha=0.7))
                self.action_texts.append(text)
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                return
                
            value_dict = q_values[s]
        
            # Check if Q-values structure is correct
            if not isinstance(value_dict, dict):
                print(f"ERROR: Q-values for ({row}, {col}) is not a dict!")
                print(f"Type: {type(value_dict)}")
                print(f"Value: {value_dict}")
                
                # Show error on board
                text = self.ax.text(s.col + 0.5, s.row + 0.5,
                                "Invalid\nQ-values",
                                fontsize=10, color='white',
                                ha='center', va='center',
                                bbox=dict(boxstyle='round', 
                                        facecolor='red', 
                                        alpha=0.8))
                self.action_texts.append(text)
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                return
            

            
            # Handle regular cells (not teleport)
            if not cell.is_teleport():
                print(f"Showing Q-values for ({row}, {col})")
                value_dict = q_values[s]
                
                # UP
                if 0 <= s.row - 1 < self.rows_no and value_dict.get('UP') is not None:
                    text = self.ax.text(s.col+0.4, s.row-1+0.75, 
                                    f"{value_dict['UP']:.1f}",
                                    fontsize=10, color='blue',
                                    ha='center', va='center')
                    self.action_texts.append(text)
                
                # DOWN
                if 0 <= s.row + 1 < self.rows_no and value_dict.get('DOWN') is not None:
                    text = self.ax.text(s.col+0.4, s.row+1+0.75, 
                                    f"{value_dict['DOWN']:.1f}",
                                    fontsize=10, color='blue',
                                    ha='center', va='center')
                    self.action_texts.append(text)
                
                # LEFT
                if 0 <= s.col - 1 < self.cols_no and value_dict.get('LEFT') is not None:
                    text = self.ax.text(s.col-1+0.4, s.row+0.75, 
                                    f"{value_dict['LEFT']:.1f}",
                                    fontsize=10, color='blue',
                                    ha='center', va='center')
                    self.action_texts.append(text)
                
                # RIGHT
                if 0 <= s.col + 1 < self.cols_no and value_dict.get('RIGHT') is not None:
                    text = self.ax.text(s.col+1+0.4, s.row+0.75, 
                                    f"{value_dict['RIGHT']:.1f}",
                                    fontsize=10, color='blue',
                                    ha='center', va='center')
                    self.action_texts.append(text)
            
            # Handle teleport cells
            elif isinstance(cell, TelCell):
                next_state = cell.get_next_cell()
                
                if next_state is not None:
                    print(f"Q-values for teleport at {s}: {q_values[s]}")
                    
                    # Show value at destination
                    text = self.ax.text(next_state.col+0.4, next_state.row+0.75,
                                    f"{q_values[s].get('UP', 0):.1f}",
                                    fontsize=10, color='magenta',
                                    ha='center', va='center')
                    self.action_texts.append(text)
                # Show Q-values at destination
                for action_name in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                    q_val = value_dict.get(action_name)
                    if q_val is not None:
                        symbol = {'UP': '↑', 'DOWN': '↓', 'LEFT': '←', 'RIGHT': '→'}[action_name]
                        text = self.ax.text(next_state.col + 0.4, next_state.row + 0.75,
                                          f"{symbol}{q_val:.1f}",
                                          fontsize=9, color='magenta',
                                          ha='center', va='center',
                                          weight='bold',
                                          bbox=dict(boxstyle='round,pad=0.3',
                                                  facecolor='pink',
                                                  alpha=0.8))
                        self.action_texts.append(text)
            
            # Update display
            self.ax.figure.canvas.draw()
            self.ax.figure.canvas.flush_events()
            
        except Exception as e:
            print(f"Error in draw_q_values: {e}")
            import traceback
            traceback.print_exc()
            
            # Show error on board
            text = self.ax.text(s.col + 0.5, s.row + 0.5,
                            f"ERROR\n{type(e).__name__}",
                            fontsize=10, color='white',
                            ha='center', va='center',
                            bbox=dict(boxstyle='round', 
                                    facecolor='red', 
                                    alpha=0.9))
            self.action_texts.append(text)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
    
    def draw_agent(self, pos=(0,0), avatar="*"):
        row, col = pos
        text = self.ax.text(col+0.4, row+0.6, avatar, fontweight="bold", color="orange", fontsize=30)
        text.set_zorder(10)
        self.moves.append(text) #moves collected all previous agent positions for visualization

        #Determines the duration of each step
        plt.pause(0.5)
        text.set_visible(False)

    def test_teleports(self):
        for row in range(board.rows_no):
            for col in range(board.cols_no):
                cell = board[row, col]
                if isinstance(cell, TelCell):
                    if cell.is_teleport():
                        dest = cell.get_next_cell()
                        print(f"Teleport at ({row},{col}) → destination ({dest.row},{dest.col})")
    

if __name__ == "__main__":
    board = MazeBoard(10, 10)
    board.draw_board()
    board.draw_agent((2,2))
    board.draw_values({Position(2,2): 5.0, Position(3,3): -2.0})
    q_values = {
        Position(2,2): {'UP': 1.0, 'DOWN': 0.5, 'LEFT': -1.0, 'RIGHT': 2.0},
        Position(3,3): {'UP': -0.5, 'DOWN': 1.5, 'LEFT': 0.0, 'RIGHT': -2.0}
    }
    board.draw_q_values(q_values)
    plt.show()
    board.test_teleports()
    

