import tensorflow as tf
import pandas as pd
from typing import Union
import random

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
        self.seen = set()

    def get_puzzles(self):
        """Get the puzzles as a tensor of shape (n, 9, 9).

        Args:
            None

        Returns:
            A tensor of shape (n, 9, 9) representing the puzzles with some numbers replaced with zeros.
        """
        return self.puzzles

    def replace_with_zeros(self, tensor: tf.Tensor, difficulty: int) -> tf.Tensor:
        """Convert a puzzle string to a tensor of shape (9, 9).

        Args:
            puzzle_str: A string of length 81 representing a sudoku puzzle.

        Returns:
            A tensor of shape (9, 9) representing the puzzle as a matrix of numbers.
        """
        indices = tf.random.shuffle(range(tensor.shape[1]), difficulty)
        return tf.tensor_scatter_nd_update(tensor, tf.reshape(tf.stack([tf.zeros_like(indices), indices], axis=1), (-1, 1)), tf.zeros((difficulty,), dtype=tf.int32))

    def process_data(self) -> tf.Tensor:
        df = pd.read_csv(self.data_path, header=None, names=["puzzle", "solution"])
        solutions_str = df["solution"].values
        dataset = tf.data.Dataset.from_tensor_slices(solutions_str)
        solutions = dataset.map(lambda x: tf.strings.bytes_split(x, encoding="utf-8"))
        solutions = solutions.map(lambda x: tf.strings.to_number(x, out_type=tf.int32))
        solutions_tensor = tf.concat([t[tf.newaxis] for t in solutions], axis=0)
        return self.replace_with_zeros(solutions_tensor, self.difficulty)

    def puzzle_string_to_tensor(self, puzzle_str: str) -> tf.Tensor:
        """Convert a puzzle string to a tensor of shape (9, 9).

        Args:
            puzzle_str: A string of length 81 representing a sudoku puzzle.

        Returns:
            A tensor of shape (9, 9) representing the puzzle as a matrix of numbers.
        """
        puzzle_chars = tf.strings.bytes_split(puzzle_str, encoding="utf-8") # split the puzzle string into a tensor of characters
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
        if tf.equal(tf.size(self.seen), tf.size(self.puzzles)):
            return None

        while True:
            index = tf.random.uniform(shape=[], minval=0, maxval=tf.size(self.puzzles), dtype=tf.int32)
            random_board = tf.gather(self.puzzles, index)

            board_tuple = tf.reshape(random_board, [-1])
            if tf.math.logical_not(tf.reduce_any(tf.equal(self.seen, board_tuple))):
                self.seen = tf.concat([self.seen, board_tuple], axis=0)
                return random_board