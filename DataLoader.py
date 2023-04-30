import tensorflow as tf
import pandas as pd
from typing import Union
from QLearningAgent import print_debug_message

class DataLoader():
    # str
    def __init__(self, data_path: str, difficulty: int):
        """Initialize the data loader with the data path and the difficulty.

        Args:
            data_path: A string representing the path to the CSV file containing the puzzles and solutions.
            difficulty: An integer indicating how many numbers to replace with zeros in each puzzle.

        Returns:
            None
        """    
        self.data_path = data_path
        self.difficulty = difficulty
        self.puzzles = self.process_data()
        self.seen = tf.zeros([0, 81], dtype=tf.int32)

    def get_puzzles(self):
        """Get the puzzles as a tensor of shape (n, 9, 9).

        Args:
            None

        Returns:
            A tensor of shape (n, 9, 9) representing the puzzles with some numbers replaced with zeros.
        """
        return self.puzzles

    def replace_with_zeros(self, tensor: tf.Tensor, difficulty: int) -> tf.Tensor:
        tensor_shape = tensor.shape.as_list()
        for i in range(tensor_shape[0]):
            indices = tf.random.shuffle(tf.range(tensor_shape[1] * tensor_shape[2]))[:difficulty]
            updates = tf.zeros((difficulty,), dtype=tf.int32)
            tensor = tf.tensor_scatter_nd_update(tensor, tf.stack([tf.repeat(i, difficulty), indices // 9, indices % 9], axis=1), updates)
        return tensor

    def process_data(self) -> tf.Tensor:
        df = pd.read_csv(self.data_path, header=None, names=["puzzle", "solution"])
        solutions_str = df["solution"].values
        dataset = tf.data.Dataset.from_tensor_slices(solutions_str)
        #solutions = dataset.map(lambda x: tf.strings.bytes_split(x, encoding="utf-8"))
        solutions = dataset.map(lambda x: tf.strings.bytes_split(x))
        solutions = solutions.map(lambda x: tf.strings.to_number(x, out_type=tf.int32))
        solutions_tensor = tf.concat([t[tf.newaxis] for t in solutions], axis=0)
        solutions_tensor = tf.reshape(solutions_tensor, (-1, 9, 9))
        print("Puzzles tensor shape:", solutions_tensor.shape)
        return self.replace_with_zeros(solutions_tensor, self.difficulty)

    def puzzle_string_to_tensor(self, puzzle_str: str) -> tf.Tensor:
        """Convert a puzzle string to a tensor of shape (9, 9).

        Args:
            puzzle_str: A string of length 81 representing a sudoku puzzle.

        Returns:
            A tensor of shape (9, 9) representing the puzzle as a matrix of numbers.
        """
        puzzle_chars = tf.strings.bytes_split(puzzle_str) # split the puzzle string into a tensor of characters
        puzzle_nums = tf.strings.to_number(puzzle_chars, out_type=tf.int32) # convert the characters to numbers
        # reshape the tensor to a 9x9 matrix
        puzzle_tensor = tf.reshape(puzzle_nums, (9, 9))
        return puzzle_tensor

    def get_random_board(self) -> Union[tf.Tensor, None]:
        """Get a random board from the puzzles without repetition.

        Args:
            None

        Returns:
            A tensor of shape (9, 9) representing a random puzzle, or None if all puzzles have been seen.
        """

        if self.seen.shape[0] == self.puzzles.shape[0]:
            return None

        attempt_count = 0
        while True:
            attempt_count += 1
            index = tf.random.uniform(shape=[], minval=0, maxval=self.puzzles.shape[0], dtype=tf.int32)
            #print_debug_message(f"Random index: {index}")
            random_board = tf.gather(self.puzzles, index)
            #print_debug_message(f"Random board (attempt {attempt_count}): \n{random_board}")
            
            board_tuple = tf.reshape(random_board, [-1])
            #print_debug_message(f"Seen tensor: {self.seen}")
            #print_debug_message(f"Board tuple: {board_tuple}")
            #print_debug_message(f"Equality result: {tf.equal(self.seen, board_tuple)}")
            #print_debug_message(f"Reduce any result: {tf.reduce_any(tf.equal(self.seen, board_tuple))}")

            if not tf.reduce_any(tf.reduce_all(tf.equal(self.seen, board_tuple), axis=1)):
                #print_debug_message(f"Seen tensor shape: {self.seen.shape}")
                #print_debug_message(f"Board tuple shape: {board_tuple.shape}")
                self.seen = tf.concat([self.seen, tf.expand_dims(board_tuple, axis=0)], axis=0)
                #print_debug_message(f"Found unique puzzle (attempt {attempt_count})")
                return random_board


