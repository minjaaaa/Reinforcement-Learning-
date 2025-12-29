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
        self.mouse_row = None
        self.mouse_col = None
        
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
        x, y = event.xdata, event.ydata

        if x is not None and y is not None:
            self.mouse_row = math.floor(y)
            self.mouse_col = math.floor(x)
    
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
            return TelCell(Position(random.randint(0, rows_no-1), random.randint(0, cols_no-1)))
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

                    self.ax.text(j+0.1, i+0.2, str(teleport_index), color="pink", fontweight="bold", fontsize=10)
                    #print(f"Teleport {teleport_index}: {i}, {j}")
                    next_cell = cell.get_next_cell()
                    self.ax.text(next_cell.col+0.1, next_cell.row+0.2, str(teleport_index),color="pink",  fontweight="bold", fontsize=10)
                    #print(f"Teleport to {teleport_index}: {next_cell.row}, {next_cell.col}")

                    teleport_index += 1  
                else:
                    board_img[i, j, :] = [255, 255, 0] # Terminal cell, YELLOW
        self.ax.imshow(board_img, extent=(0, self.cols_no, self.rows_no, 0), origin="upper")

    def hide_text(self, text):
        text.set_visible(False)
    
    def show_text(self, text):
        text.set_visible(True)

    def draw_values(self, values: dict[Position , float]):
        if len(self.value_texts):
            [self.hide_text(text) for text in self.value_texts]
        else:
            for s in values:                
                text = self.ax.text(s.col+0.4, s.row+0.75, str(f"{values[s]:.1f}"), fontsize=10)
                self.value_texts.append(text)
    
    def draw_q_values(self , q_values):
        [self.hide_text(text) for text in self.value_texts]
        [self.hide_text(text) for text in self.action_texts]
        
        #if mouse is out of bounds, set to (0,0)
        row = self.mouse_row if self.mouse_row is not None else 0
        col = self.mouse_col if self.mouse_col is not None else 0
        s = Position(row, col)

        try:
            #if the cell is not a teleport cell and has q_values
            if s in q_values and not self[s.row, s.col].is_teleport():
                value_dict = q_values[s]
                #chwcking boundings and if the action has a value
                if 0<=s.row-1<self.rows_no and value_dict['UP'] is not None:
                    text = self.ax.text(s.col+0.4, s.row-1+0.75, str(f"{value_dict['UP']:.1f}"), fontsize=10)
                    if text not in self.action_texts:
                        self.action_texts.append(text)
                if 0<=s.row+1<self.rows_no and value_dict['DOWN'] is not None:
                    text = self.ax.text(s.col+0.4, s.row+1+0.75, str(f"{value_dict['DOWN']:.1f}"), fontsize=10)
                    if text not in self.action_texts:
                        self.action_texts.append(text)
                if 0<=s.col-1< self.cols_no and value_dict['LEFT'] is not None:
                    text = self.ax.text(s.col-1+0.4, s.row+0.75, str(f"{value_dict['LEFT']:.1f}"), fontsize=10)
                    if text not in self.action_texts:
                        self.action_texts.append(text)
                if 0<=s.col+1<self.cols_no and value_dict['RIGHT'] is not None:
                    text = self.ax.text(s.col+1+0.4, s.row+0.75, str(f"{value_dict['RIGHT']:.1f}"), fontsize=10)
                    if text not in self.action_texts:
                        self.action_texts.append(text)
            elif self[s.row, s.col].is_teleport():
                cell = self[s.row, s.col]
                if isinstance(cell, TelCell):
                    next_state = cell.get_next_cell()
                    print(f"Q_values: {q_values[s]}")
                    text = self.ax.text(next_state.col+0.4, next_state.row+0.75, str(f"{q_values[s]['UP']:.1f}"), fontsize=10)
                    if text not in self.action_texts:
                        self.action_texts.append(text)
        except Exception as e:
            print(e)
    
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
    

