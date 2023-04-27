import argparse
import tensorflow as tf
import SudokuTrainer
import QLearningAgent
import SudokuEnvironment
import DataLoader

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
    tpu_strategy = tf.distribute.cluster_resolver.TPUClusterResolver()
    decay_steps = 1000           # Number of steps before applying the exploration rate decay
    max_memory_size = 5000       # Maximum size of the experience replay memory
    file_path = "/weights"  # File path for saving and loading the Q-Network model

    data_loader_easy = DataLoader("/resources", 8)
    data_loader_medium = DataLoader("/resourceS", 16)
    #data_loader_medium_hard = DataLoader("/resources", 24)
    #data_loader_hard = DataLoader("/resources", 32)
    #data_loader_very_hard = DataLoader("/resources", 40)

    easy_puzzles = data_loader_easy.get_puzzles()
    #medium_puzzles = data_loader_medium.get_puzzles()
    #medium_hard_puzzles = data_loader_medium_hard.get_puzzles()
    #hard_puzzles = data_loader_hard.get_puzzles()
    #very_hard_puzzles = data_loader_very_hard.get_puzzles()

    env = SudokuEnvironment(easy_puzzles) # pass the list of puzzles to the environment
    agent = QLearningAgent(learning_rate, discount_factor, exploration_rate, exploration_decay, tpu_strategy, decay_steps, max_memory_size, file_path)

    trainer_easy = SudokuTrainer(agent, env, data_loader_easy)
    trainer_easy.train(10, 100, 1024, 100)