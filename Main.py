import argparse
import tensorflow as tf
import SudokuTrainer
import QLearningAgent
import SudokuEnvironment
import DataLoader

def __main__():
    learning_rate = 0.1          # The learning rate for the Q-Learning algorithm
    discount_factor = 0.99       # The discount factor for future rewards
    exploration_rate = 1.0       # Initial exploration rate (epsilon) for the epsilon-greedy strategy
    exploration_decay = 0.995    # Exploration rate decay factor
    tpu_strategy = tf.distribute.cluster_resolver.TPUClusterResolver()  # Replace TPU_IP_ADDRESS with your TPU's IP address
    decay_steps = 1000           # Number of steps before applying the exploration rate decay
    max_memory_size = 5000       # Maximum size of the experience replay memory
    file_path = "/weights"  # File path for saving and loading the Q-Network model

    data_loader = DataLoader("/resources", 10)
    puzzles = data_loader.get_puzzles()
    env = SudokuEnvironment(puzzles) # pass the list of puzzles to the environment
    agent = QLearningAgent(learning_rate, discount_factor, exploration_rate, exploration_decay, tpu_strategy, decay_steps, max_memory_size, file_path)

    trainer = SudokuTrainer(agent, env, data_loader)
    trainer.train(10, 100, 1024, 100)

def parse_arguments():
    pass

def setup_tpu_strategy():
    pass # return tf.distribute.TPUStrategy