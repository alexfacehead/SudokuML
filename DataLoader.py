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
        indices = random.sample(range(tensor.shape[1]), difficulty)
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
        puzzle_list = [int(c) for c in puzzle_str]
        puzzle_tensor = tf.reshape(tf.constant(puzzle_list, dtype=tf.int32), (9, 9))
        return puzzle_tensor

    def get_random_board(self) -> Union[tf.Tensor, None]:
        if len(self.seen) == len(self.puzzles):
            return None

        while True:
            index = tf.random.uniform(shape=[], minval=0, maxval=len(self.puzzles), dtype=tf.int32)
            random_board = self.puzzles[index]

            board_tuple = tuple(random_board.numpy().flatten())
            if board_tuple not in self.seen:
                self.seen.add(board_tuple)
                return random_board