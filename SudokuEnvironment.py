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
        
        self.REWARD_DICT = {
        "invalid_move": -10,
        "valid_move": 10,
        "row_col_box_completed": 5,
        "puzzle_solved": 50
        }

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
    
    def step(self, action: Tuple[int, int, int], valid_actions: List[Tuple[int, int, int]], all_available_actions: List[Tuple[int, int, int]]) -> Tuple[tf.Tensor, float, bool, bool]:
        row, col, num = action
        is_valid = self.is_valid_move(row, col, num, suppress=False)

        if self.board[row, col] != 0 or (action not in all_available_actions and action not in valid_actions):
            self.incorrect_moves_count += 1
            done = self.incorrect_moves_count >= self.max_incorrect_moves
            reward = self.get_reward(action, valid_actions)
            next_state = tf.identity(self.board)
            status = False

            return next_state, reward, done, status

        reward = self.get_reward(action, valid_actions)
        if is_valid:
            indices = tf.convert_to_tensor([[row, col]])
            updates = tf.convert_to_tensor([num])
            self.board = tf.tensor_scatter_nd_update(self.board, indices, updates)

        else:
            self.incorrect_moves_count += 1

        status = self.is_solved()
        done = self.incorrect_moves_count >= self.max_incorrect_moves or status
        if status:
            reward += self.REWARD_DICT["puzzle_solved"]
            print_debug_message("is_solved bonus: " + str(self.REWARD_DICT["puzzle_solved"]))
        next_state = tf.identity(self.board)
        #msg2 = "Step: Action: " + format_action_tuple(action) + ", Reward: " + str(reward)
        msg2 = "Done: " + str(done)
        print_debug_message(msg2)

        return next_state, reward, done, status

    def render(self):
        """Print the board to the standard output.
        """
        print(self.board)

    def is_valid_move(self, row: int, col: int, num: int, suppress=False) -> bool:
        """Check if a given move is valid or not (i.e., there is not another number in the row/column/grid area that is
        idenitcal to the number chosen by num).

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

            return False

        grid_row, grid_col = row // 3 * 3, col // 3 * 3
        grid = tf.slice(self.board, [grid_row, grid_col], [3, 3])

        if tf.reduce_any(tf.equal(grid, num)).numpy() or self.board[row, col].numpy() != 0:

            return False
        
        msg7 = "Valid move determined."
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
                box = tf.slice(self.board, [row, col], [3, 3])
                box_flat = tf.reshape(box, (-1,))
                unique_box_count = tf.reduce_sum(tf.cast(tf.math.bincount(box_flat, minlength=10)[1:] > 0, dtype=tf.int32))
                if unique_box_count != 9:
                    return False

        print_debug_message("Solved.")

        return True

    def get_valid_actions(self, board_state: tf.Tensor) -> List[Tuple[int, int, int]]:
        # Find the indices of the zeros in the board state
        zero_indices = tf.where(tf.equal(board_state, 0))

        # Define a function that takes an index and returns a list of valid actions for that index
        def get_actions_for_index(index):
            row = index[0]
            col = index[1]
            # Create a tensor of shape (9,) with numbers from 1 to 9
            nums = tf.range(1, 10)
            # Create a boolean mask of shape (9,) indicating which numbers are valid for the index
            mask = tf.map_fn(lambda num: self.is_valid_move(row, col, num, suppress=True), nums)
            # Gather the valid numbers using the mask
            valid_nums = tf.boolean_mask(nums, mask)
            # Create a tensor of shape (n, 3) with the row, col, and valid nums as actions
            actions = tf.concat([tf.cast(tf.fill([tf.shape(valid_nums)[0], 1], row), dtype=tf.int32),
                                tf.cast(tf.fill([tf.shape(valid_nums)[0], 1], col), dtype=tf.int32),
                                tf.reshape(valid_nums, [-1, 1])], axis=1)
            
            # Pad actions tensor to have a fixed shape of (9, 3)
            actions_padded = tf.pad(actions, [[0, 9 - tf.shape(actions)[0]], [0, 0]])
            return actions_padded

        # Apply the function to each zero index and stack the results into a tensor of shape (m, 9, 3)
        valid_actions = tf.map_fn(get_actions_for_index, zero_indices, dtype=tf.int32)

        # Remove padding and convert the tensor to a list of tuples
        valid_actions = tf.reshape(valid_actions, [-1, 3])
        valid_actions = valid_actions[valid_actions[:, 2] != 0]
        valid_actions_list = [tuple(action.numpy()) for action in valid_actions]
        
        return valid_actions_list


    def get_valid_actions_old(self, board_state: tf.Tensor) -> List[Tuple[int, int, int]]:
        """Get the list of available valid actions for a given board state.

        Args:
            board_state: A tensor of shape (9, 9) representing the current board state.

        Returns:
            A list of tuples of the form (row, col, num) representing the possible actions.
        """
        valid_actions = []

        for row in range(9):
            for col in range(9):
                if board_state[row, col] == 0:
                    for num in range(1, 10):
                        if self.is_valid_move(row, col, num, suppress=True):
                            valid_actions.append((row, col, num))
        print_debug_message(f"Valid actions: {valid_actions}")

        return valid_actions

    def get_all_available_actions(self, board_state: tf.Tensor) -> List[Tuple[int, int, int]]:
        """Get the list of all available actions for a given board state.

        Args:
            board_state: A tensor of shape (9, 9) representing the current board state.

        Returns:
            A list of tuples of the form (row, col, num) representing the possible actions.
        """
        # Find the indices of the zeros in the board state
        zero_indices = tf.where(tf.equal(board_state, 0))

        # Define a function that takes an index and returns a list of all actions for that index
        def get_actions_for_index(index):
            row = index[0]
            col = index[1]
            # Create a tensor of shape (9, 3) with the row, col, and numbers from 1 to 9 as actions
            actions = tf.concat([tf.cast(tf.fill([9, 1], row), dtype=tf.int32),
                                tf.cast(tf.fill([9, 1], col), dtype=tf.int32),
                                tf.reshape(tf.range(1, 10), [-1, 1])], axis=1)
            return actions

        # Apply the function to each zero index and stack the results into a tensor of shape (m, 9, 3)
        all_available_actions = tf.map_fn(get_actions_for_index, zero_indices, dtype=tf.int32)

        # Convert the tensor to a list of tuples
        all_available_actions = tf.reshape(all_available_actions, [-1, 3])
        all_available_actions_list = [tuple(action.numpy()) for action in all_available_actions]

        #print_debug_message(f"Board state:\n{board_state}")
        #print_debug_message(f"All available actions: {all_available_actions_list}")

        return all_available_actions_list

    def get_all_available_actions_old(self, board_state: tf.Tensor) -> List[Tuple[int, int, int]]:
        """Get the list of all available actions for a given board state.

        Args:
            board_state: A tensor of shape (9, 9) representing the current board state.

        Returns:
            A list of tuples of the form (row, col, num) representing the possible actions.
        """
        all_available_actions = []

        for row in range(9):
            for col in range(9):
                if board_state[row, col] == 0:
                    for num in range(1, 10):
                        all_available_actions.append((row, col, num))

        #print_debug_message(f"Board state:\n{board_state}")
        print_debug_message(f"All available actions: {all_available_actions}")

        return all_available_actions

    def get_reward(self, action: Tuple[int, int, int], valid_actions: List[Tuple[int, int, int]]) -> float:
        row, col, num = action

        if self.board[row, col] != 0:
            return self.REWARD_DICT["invalid_move"]  # Penalty for attempting to place a number in an already filled cell

        # Update the board temporarily
        temp_board = tf.tensor_scatter_nd_update(self.board, [[row, col]], [num])

        row_filled = tf.reduce_all(tf.math.equal(tf.math.reduce_sum(tf.one_hot(temp_board[row], 10), axis=0)[1:], 1))
        col_filled = tf.reduce_all(tf.math.equal(tf.math.reduce_sum(tf.one_hot(temp_board[:, col], 10), axis=0)[1:], 1))
        box_row_start = (row // 3) * 3
        box_col_start = (col // 3) * 3
        box = tf.slice(temp_board, [box_row_start, box_col_start], [3, 3])
        box_filled = tf.reduce_all(tf.math.equal(tf.math.reduce_sum(tf.one_hot(tf.reshape(box, (-1,)), 10), axis=0)[1:], 1))

        row_col_box_bonus = sum([row_filled.numpy(), col_filled.numpy(), box_filled.numpy()]) * self.REWARD_DICT["row_col_box_completed"]

        # Ensure row_col_box_bonus doesn't exceed 5
        row_col_box_bonus = min(row_col_box_bonus, 5)

        is_solved = self.is_solved()

        if action in valid_actions:
            reward = self.REWARD_DICT["valid_move"] + row_col_box_bonus
        else:
            reward = -5 + row_col_box_bonus

        if is_solved:
            reward += self.REWARD_DICT["puzzle_solved"]
            print_debug_message("is_solved bonus: " + str(self.REWARD_DICT["puzzle_solved"]))
        if row_col_box_bonus > 0:
            print_debug_message("row_col_box_bonus: " + str(row_col_box_bonus))

        return reward