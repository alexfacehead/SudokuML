from typing import List, Tuple
import numpy as np

class SudokuEnvironment():
    # np.ndarray
    def __init__(self, sudoku_board: np.ndarray):
        self.initial_board = sudoku_board.copy()
        self.board = sudoku_board.copy()

    def reset(self) -> np.ndarray:
        self.board = self.initial_board.copy()
        return self.board

    # Tuple[int, int, int]
    def step(self, action: Tuple[int, int, int]) -> Tuple[np.ndarray, float, bool]:
        row, col, num = action

        if self.is_valid_move(row, col, num):
            self.board[row, col] = num
            done = self.is_solved()
        else:
            done = False

        reward = self.get_reward(action)
        next_state = self.board.copy()
        return next_state, reward, done

    def render(self):
        print(self.board)

    def is_valid_move(self, row: int, col: int, num: int) -> bool:
        # Check row and column
        if np.any(self.board[row, :] == num) or np.any(self.board[:, col] == num):
            return False

        # Check 3x3 grid
        grid_row, grid_col = row // 3 * 3, col // 3 * 3
        if np.any(self.board[grid_row:grid_row + 3, grid_col:grid_col + 3] == num):
            return False

        return True

    def is_solved(self) -> bool:
        for row in range(9):
            if len(set(self.board[row])) != 9:
                return False
            
        for col in range(9):
            if len(set(self.board[:, col])) != 9:
                return False
                
        for row in range(0, 9, 3):
            for col in range(0, 9, 3):
                box = [self.board[row + i, col + j] for i in range(3) for j in range(3)]
                if len(set(box)) != 9:
                    return False
                    
        return True

    def get_available_actions(self, board_state: np.ndarray) -> List[Tuple[int, int, int]]:
        available_actions = []

        for row in range(9):
            for col in range(9):
                if board_state[row, col] == 0:
                    for num in range(1, 10):
                        if self.is_valid_move(row, col, num):
                            available_actions.append((row, col, num))

        return available_actions
    
    def get_reward(self, action: Tuple[int, int, int]) -> float:
        row, col, num = action
        if self.is_valid_move(row, col, num):
            grid_copy = copy.deepcopy(self.board)
            grid_copy[row, col] = num
            row_filled = len(set(grid_copy[row])) == 9
            col_filled = len(set(grid_copy[:, col])) == 9
            
            box_row_start = (row // 3) * 3
            box_col_start = (col // 3) * 3
            box = [grid_copy[box_row_start + i, box_col_start + j] for i in range(3) for j in range(3)]
            box_filled = len(set(box)) == 9
            
            bonus = (row_filled + col_filled + box_filled) * 2
            return 1 + bonus
        else:
            return -5

    #def get_reward_skeleton(self, action: Tuple[int, int, int]):
        # action is a tuple of (row, column, number)
        # return a reward based on the action and the current state of the grid
        # for example:
        # if the action is valid and fills an empty cell, return 1
        # if the action is invalid or overwrites an existing number, return -1
        # if the action completes the puzzle, return 10
        # you can adjust these values as you like
       #return 0.0 # return float