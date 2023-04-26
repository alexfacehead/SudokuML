from typing import List, Tuple
import tensorflow as tf

class SudokuEnvironment():
    # tensor
    def __init__(self, sudoku_boards: List[tf.Tensor], max_incorrect_moves: int=5):
        self.sudoku_boards = sudoku_boards # store the list of puzzles
        self.board = None # initialize the board as None
        self.max_incorrect_moves = max_incorrect_moves
        self.incorrect_moves_count = 0

    def reset(self) -> tf.Tensor:
        # convert the list of sudoku boards to a tensor of shape (n, 9, 9)
        self.sudoku_boards = tf.convert_to_tensor(self.sudoku_boards)
        # choose a random index from 0 to n-1
        index = tf.random.uniform((), minval=0, maxval=tf.shape(self.sudoku_boards)[0], dtype=tf.int32)
        # select the board at that index
        self.board = self.sudoku_boards[index]
        self.incorrect_moves_count = 0
        return self.board
    
    # Tuple[int, int, int]
    def step(self, action: Tuple[int, int, int]) -> Tuple[tf.Tensor, float, bool]:
        row, col, num = action

        if self.is_valid_move(row, col, num):
            indices = tf.convert_to_tensor([[row, col]])
            updates = tf.convert_to_tensor([num])
            self.board = tf.tensor_scatter_nd_update(self.board, indices, updates) # Move this inside the if block
            done = self.is_solved()
        else:
            self.incorrect_moves_count += 1
            done = self.incorrect_moves_count >= self.max_incorrect_moves

        reward = self.get_reward(action)
        next_state = tf.identity(self.board)
        return next_state, reward, done

    def render(self):
        print(self.board)

    def is_valid_move(self, row: int, col: int, num: int) -> bool:
        row_values = tf.slice(self.board, [row, 0], [1, 9])
        col_values = tf.slice(self.board, [0, col], [9, 1])
        
        if tf.reduce_any(row_values == num) or tf.reduce_any(col_values == num):
            return False

        grid_row, grid_col = row // 3 * 3, col // 3 * 3
        grid = tf.slice(self.board, [grid_row, grid_col], [3, 3])
        
        if tf.reduce_any(grid == num):
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

    def get_available_actions(self, board_state: tf.Tensor) -> List[Tuple[int, int, int]]:
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
            temp_board = tf.tensor_scatter_nd_update(self.board, [[row, col]], [num])

            row_filled = tf.reduce_all(tf.math.equal(tf.math.reduce_sum(tf.one_hot(temp_board[row], 10), axis=0)[1:], 1))
            col_filled = tf.reduce_all(tf.math.equal(tf.math.reduce_sum(tf.one_hot(temp_board[:, col], 10), axis=0)[1:], 1))

            box_row_start = (row // 3) * 3
            box_col_start = (col // 3) * 3
            box = tf.slice(temp_board, [box_row_start, box_col_start], [3, 3])
            box_filled = tf.reduce_all(tf.math.equal(tf.math.reduce_sum(tf.one_hot(tf.reshape(box, (-1,)), 10), axis=0)[1:], 1))

            bonus = tf.reduce_sum(tf.stack([row_filled, col_filled, box_filled])) * 2

            return 1 + int(bonus)
        else:
            return -5
