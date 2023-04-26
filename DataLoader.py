import tensorflow as tf
import pandas as pd
from typing import Union
import random

class DataLoader():
    # str
    def __init__(self, data_path: str, difficulty: int):
        self.data_path = data_path
        self.difficulty = difficulty
        self.puzzles = self.process_data()
        self.seen = set()

    def get_puzzles(self):
        return self.puzzles

    def replace_with_zeros(self, tensor: tf.Tensor, difficulty: int) -> tf.Tensor:
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
        puzzle_chars = tf.strings.bytes_split(puzzle_str, encoding="utf-8") # split the puzzle string into a tensor of characters
        puzzle_nums = tf.strings.to_number(puzzle_chars, out_type=tf.int32) # convert the characters to numbers
        # reshape the tensor to a 9x9 matrix
        puzzle_tensor = tf.reshape(puzzle_nums, (9, 9))
        return puzzle_tensor

    def get_random_board(self) -> Union[tf.Tensor, None]:
        if tf.equal(tf.size(self.seen), tf.size(self.puzzles)):
            return None

        while True:
            index = tf.random.uniform(shape=[], minval=0, maxval=tf.size(self.puzzles), dtype=tf.int32)
            random_board = tf.gather(self.puzzles, index)

            board_tuple = tf.reshape(random_board, [-1])
            if tf.math.logical_not(tf.reduce_any(tf.equal(self.seen, board_tuple))):
                self.seen = tf.concat([self.seen, board_tuple], axis=0)
                return random_board