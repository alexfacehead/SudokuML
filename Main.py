import os
import tensorflow as tf
from SudokuTrainer import SudokuTrainer
from QLearningAgent import QLearningAgent
from SudokuEnvironment import SudokuEnvironment
from DataLoader import DataLoader
from DataLoader import DataLoader

def __main__():
    """Run the main program.

    Args:
        None

    Returns:
        None
    """
    learning_rate = 0.1          # The learning rate for the Q-Learning algorithm
    discount_factor = 0.99       # The discount factor for future rewards
    exploration_rate = 1.0       # Initial exploration rate (epsilon) for the epsilon-greedy strategy
    exploration_decay = 0.995    # Exploration rate decay factor

    # Initialize the TPU strategy
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    tpu_strategy = tf.distribute.TPUStrategy(resolver)

    decay_steps = 1000           # Number of steps before applying the exploration rate decay
    max_memory_size = 5000       # Maximum size of the experience replay memory
    file_path = "/weights"  # File path for saving and loading the Q-Network model

    data_loader_easy = DataLoader("./resources/sudoku_mini_easy.csv", 8)

    easy_puzzles = data_loader_easy.get_puzzles()

    env = SudokuEnvironment(easy_puzzles) # pass the list of puzzles to the environment
    agent = QLearningAgent(learning_rate, discount_factor, exploration_rate, exploration_decay, tpu_strategy, decay_steps, max_memory_size, file_path)

    trainer_easy = SudokuTrainer(agent, env, data_loader_easy)
    trainer_easy.train(10, 100, 1024, 100)

__main__()