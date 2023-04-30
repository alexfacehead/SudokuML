from typing import List, Tuple, Optional
import tensorflow as tf
from QLearningAgent import print_debug_message
from QLearningAgent import format_action_tuple

class SudokuEnvironment():
    # tensor
    def __init__(self, sudoku_boards: List[tf.Tensor], max_incorrect_moves: int=5):
        """Initialize the sudoku environment with the list of puzzles and the maximum number of incorrect moves.

        Args:
            sudoku_boards: A list of tensors of shape (9, 9) representing the puzzles with some numbers replaced with zeros.
            max_incorrect_moves: An integer indicating the maximum number of incorrect moves allowed before terminating the episode.

        Returns:
            None
        """
        self.sudoku_boards = sudoku_boards # store the list of puzzles
        self.board = None # initialize the board as None
        self.max_incorrect_moves = max_incorrect_moves
        self.incorrect_moves_count = 0
        print("Max incorrect = " + str(self.max_incorrect_moves))

    def reset(self, sudoku_board: Optional[tf.Tensor] = None) -> tf.Tensor:
        """Reset the environment and use the provided puzzle or choose a random one if not provided.

        Args:
            sudoku_board: An optional tensor of shape (9, 9) representing a Sudoku puzzle.

        Returns:
            A tensor of shape (9, 9) representing the chosen puzzle.
        """
        if sudoku_board is None:
            # convert the list of sudoku boards to a tensor of shape (n, 9, 9)
            self.sudoku_boards = tf.convert_to_tensor(self.sudoku_boards)
            # choose a random index from 0 to n-1
            index = tf.random.uniform((), minval=0, maxval=tf.shape(self.sudoku_boards)[0], dtype=tf.int32)
            # select the board at that index
            self.board = self.sudoku_boards[index]
        else:
            self.board = sudoku_board

        self.incorrect_moves_count = 0
        return self.board

    def step_old(self, action: Tuple[int, int, int]) -> Tuple[tf.Tensor, float, bool]:
        """Take an action and observe the next state and reward.

        Args:
            action: A tuple of the form (row, col, num) representing the action to take.

        Returns:
            A tuple of the form (next_state: tf.Tensor,
                                reward: float,
                                done: bool)
            representing the next board state,
            the reward for taking the action,
            and whether the episode is over or not.
        """
        row, col, num = action
        print("Calling is_valid_move from step")
        is_valid = self.is_valid_move(row, col, num, suppress=True)

        # Calculate the reward first
        reward = self.get_reward(action)

        if is_valid:
            indices = tf.convert_to_tensor([[row, col]])
            updates = tf.convert_to_tensor([num])
            self.board = tf.tensor_scatter_nd_update(self.board, indices, updates)  # Move this inside the if block
            done = self.is_solved()
            msg = "Board solved? " + str(done)
            print_debug_message(msg)
            print(msg)
        else:
            self.incorrect_moves_count += 1
            done = self.incorrect_moves_count >= self.max_incorrect_moves

        next_state = tf.identity(self.board)
        return next_state, reward, done


    # Tuple[int, int, int]
    def step(self, action: Tuple[int, int, int]) -> Tuple[tf.Tensor, float, bool]:
        """Take an action and observe the next state and reward.

        Args:
            action: A tuple of the form (row, col, num) representing the action to take.

        Returns:
            A tuple of the form (next_state: tf.Tensor,
                                reward: float,
                                done: bool)
            representing the next board state,
            the reward for taking the action,
            and whether the episode is over or not.
        """
        row, col, num = action
        print("Calling is_valid_move from step")
        if self.is_valid_move(row, col, num, suppress=True):
            indices = tf.convert_to_tensor([[row, col]])
            updates = tf.convert_to_tensor([num])
            self.board = tf.tensor_scatter_nd_update(self.board, indices, updates) # Move this inside the if block
            done = self.is_solved()
            msg = "Board solved? " + str(done)
            print_debug_message(msg)
            print(msg)
        else:
            self.incorrect_moves_count += 1
            done = self.incorrect_moves_count >= self.max_incorrect_moves

        reward = self.get_reward(action)
        next_state = tf.identity(self.board)
        return next_state, reward, done

    def render(self):
        """Print the board to the standard output.
        """
        print(self.board)

    def is_valid_move(self, row: int, col: int, num: int, suppress=False) -> bool:
        """Check if a given move is valid or not.

        Args:
            row: An integer indicating the row index of the move.
            col: An integer indicating the column index of the move.
            num: An integer indicating the number to place in the move.

        Returns:
            A boolean indicating whether the move is valid or not.
        """
        row_values = tf.slice(self.board, [row, 0], [1, 9])
        col_values = tf.slice(self.board, [0, col], [9, 1])

        if tf.reduce_any(tf.equal(row_values, num)).numpy() or tf.reduce_any(tf.equal(col_values, num)).numpy():
            print_debug_message("Invalid move")
            return False

        grid_row, grid_col = row // 3 * 3, col // 3 * 3
        grid = tf.slice(self.board, [grid_row, grid_col], [3, 3])

        if tf.reduce_any(tf.equal(grid, num)).numpy() or self.board[row, col].numpy() != 0:
            print_debug_message("Invalid move")
            return False

        # Debug output messages
        #row_values_debug_msg = "Row values: " + str(row_values) + ", Expected: " + str(tf.slice(self.board, [row, 0], [1, 9]))
        #print_debug_message(row_values_debug_msg)

        #col_values_debug_msg = "Column values: " + str(col_values) + ", Expected: " + str(tf.slice(self.board, [0, col], [9, 1]))
        #print_debug_message(col_values_debug_msg)

        #grid_values_debug_msg = "Grid values: " + str(grid) + ", Expected: " + str(tf.slice(self.board, [grid_row, grid_col], [3, 3]))
        #print_debug_message(grid_values_debug_msg)

        msg7 = "Valid move made.\n"
        if not suppress:
            print_debug_message(msg7)
        return True

    
    def is_solved(self) -> bool:
        """Check if the board is solved or not.

        Args:
            None

        Returns:
            A boolean indicating whether the board is solved or not.
        """
        for row in range(9):
            unique_row_count = tf.reduce_sum(tf.cast(tf.math.bincount(self.board[row], minlength=10)[1:] > 0, dtype=tf.int32))
            if unique_row_count != 9:
                return False
                    
        for col in range(9):
            unique_col_count = tf.reduce_sum(tf.cast(tf.math.bincount(self.board[:, col], minlength=10)[1:] > 0, dtype=tf.int32))
            if unique_col_count != 9:
                return False
                        
        for row in range(0, 9, 3):
            for col in range(0, 9, 3):
                box = [self.board[row + i, col + j] for i in range(3) for j in range(3)]
                unique_box_count = tf.reduce_sum(tf.cast(tf.math.bincount(tf.reshape(box, (-1,)), minlength=10)[1:] > 0, dtype=tf.int32))
                if unique_box_count != 9:
                    return False

        #print_debug_message("Solved.\n")
        return True

    def get_available_actions(self, board_state: tf.Tensor) -> List[Tuple[int, int, int]]:
        """Get the list of available actions for a given board state.

        Args:
            board_state: A tensor of shape (9, 9) representing the current board state.

        Returns:
            A list of tuples of the form (row, col, num) representing the possible actions.
        """
        available_actions = []

        for row in range(9):
            for col in range(9):
                if board_state[row, col] == 0:
                    for num in range(1, 10):
                        if self.is_valid_move(row, col, num, suppress=True):
                            available_actions.append((row, col, num))
        print_debug_message(f"Board state:\n{board_state}")
        print_debug_message(f"Available actions: {available_actions}")
        return available_actions
    
    def get_reward(self, action: Tuple[int, int, int]) -> float:
        row, col, num = action
        is_valid = self.is_valid_move(row, col, num)
        msg = "Calling is_valid_move from get_reward on " + str(format_action_tuple(action)) + "and " + str(is_valid)
        print_debug_message(msg)
        if is_valid:
            print_debug_message("Is valid move! Heading into reward block")
            print_debug_message(f"Before update: {self.board.numpy()}")
            temp_board = tf.tensor_scatter_nd_update(self.board, [[row, col]], [num])
            print_debug_message(f"After update: {temp_board.numpy()}")

            row_filled = tf.reduce_all(tf.math.equal(tf.math.reduce_sum(tf.one_hot(temp_board[row], 10), axis=0)[1:], 1))
            col_filled = tf.reduce_all(tf.math.equal(tf.math.reduce_sum(tf.one_hot(temp_board[:, col], 10), axis=0)[1:], 1))

            box_row_start = (row // 3) * 3
            box_col_start = (col // 3) * 3
            box = tf.slice(temp_board, [box_row_start, box_col_start], [3, 3])
            box_filled = tf.reduce_all(tf.math.equal(tf.math.reduce_sum(tf.one_hot(tf.reshape(box, (-1,)), 10), axis=0)[1:], 1))

            print_debug_message(f"Row filled: {row_filled.numpy()}")
            print_debug_message(f"Col filled: {col_filled.numpy()}")
            print_debug_message(f"Box filled: {box_filled.numpy()}")

            # Convert boolean tensor to int32 tensor before passing it to tf.reduce_sum()
            bonus = tf.reduce_sum(tf.stack([tf.cast(row_filled, tf.int32), tf.cast(col_filled, tf.int32), tf.cast(box_filled, tf.int32)])) * 2

            return 1 + int(bonus)
        else:
            return -5


    def get_reward_old(self, action: Tuple[int, int, int]) -> float:
        row, col, num = action
        is_valid = self.is_valid_move(row, col, num)
        msg = "Calling is_valid_move from get_reward on " + str(format_action_tuple(action)) + "and " + str(is_valid)
        print_debug_message(msg)
        if is_valid:
            print_debug_message("Is valid move! Heading into reward block")
            print_debug_message(f"Before update: {self.board.numpy()}")
            temp_board = tf.tensor_scatter_nd_update(self.board, [[row, col]], [num])
            print_debug_message(f"After update: {temp_board.numpy()}")

            row_filled = tf.reduce_all(tf.math.equal(tf.math.reduce_sum(tf.one_hot(temp_board[row], 10), axis=0)[1:], 1))
            col_filled = tf.reduce_all(tf.math.equal(tf.math.reduce_sum(tf.one_hot(temp_board[:, col], 10), axis=0)[1:], 1))

            box_row_start = (row // 3) * 3
            box_col_start = (col // 3) * 3
            box = tf.slice(temp_board, [box_row_start, box_col_start], [3, 3])
            box_filled = tf.reduce_all(tf.math.equal(tf.math.reduce_sum(tf.one_hot(tf.reshape(box, (-1,)), 10), axis=0)[1:], 1))

            print_debug_message(f"Row filled: {row_filled.numpy()}")
            print_debug_message(f"Col filled: {col_filled.numpy()}")
            print_debug_message(f"Box filled: {box_filled.numpy()}")

            bonus = tf.reduce_sum(tf.stack([row_filled, col_filled, box_filled])) * 2

            return 1 + int(bonus)
        else:
            return -5