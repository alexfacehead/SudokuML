import DataLoader
import QLearningAgent
import SudokuEnvironment
import numpy as np

class SudokuTrainer():
    def __init__(self, agent: QLearningAgent, environment: SudokuEnvironment, data_loader: DataLoader):
        pass

    def train(self, episodes: int, batch_size: int, target_update_interval: int):
        pass

    def evaluate(self, episodes: int):
        pass # return -> float

    def play(self, sudoku_board: np.ndarray):
        pass # return -> np.ndarray

    def save_training_history():
        pass # write to "history.log"