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
    try:
        tpu = None
        if 'COLAB_TPU_ADDR' in os.environ:
            tpu = 'grpc://' + os.environ['COLAB_TPU_ADDR']
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu)
            tf.config.experimental_connect_to_cluster(resolver)
            tf.tpu.experimental.initialize_tpu_system(resolver)
            strategy = tf.distribute.TPUStrategy(resolver)
        else:
            strategy = tf.distribute.OneDeviceStrategy("GPU:0")
    except ValueError:
        strategy = tf.distribute.OneDeviceStrategy("CPU:0")

    decay_steps = 1000           # Number of steps before applying the exploration rate decay
    max_memory_size = 5000       # Maximum size of the experience replay memory

    file_path = "/home/dev/SudokuML/weights"  # File path for saving and loading the Q-Network model

    if os.path.exists('/content/drive'):  # Google Colab environment
        file_path = '/content/drive/MyDrive/SudokuML/weights'
    else:  # Local environment
        file_path = '/home/dev/SudokuML/weights'

    data_loader_easy = DataLoader("./resources/sudoku_mini_easy.csv", 8)

    easy_puzzles = data_loader_easy.get_puzzles()

    with strategy.scope():
        env = SudokuEnvironment(easy_puzzles) # pass the list of puzzles to the environment
        agent = QLearningAgent(learning_rate, discount_factor, exploration_rate, exploration_decay, strategy, decay_steps, max_memory_size, file_path)

        trainer_easy = SudokuTrainer(agent, env, data_loader_easy)
        trainer_easy.train(10, 100, 1024, 100)

if __name__ == "__main__":
    __main__()