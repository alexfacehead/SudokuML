import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from SudokuTrainer import SudokuTrainer
from QLearningAgent import QLearningAgent
from SudokuEnvironment import SudokuEnvironment
from DataLoader import DataLoader
from DataLoader import DataLoader
import argparse
from dotenv import load_dotenv

load_dotenv()
google_colab_path = os.getenv("google_colab_path")
local_path = os.getenv("local_path")

def __main__():
    """Run the main program.

    Args:
        None

    Returns:
        None
    """
    # Argparse stuff
    parser = argparse.ArgumentParser(description="Sudoku reinforcement learning")
    parser.add_argument("--fresh", action="store_true", help="Delete debug_output.txt if it exists")
    args = parser.parse_args()

    if args.fresh and os.path.exists("debug_output.txt"):
        os.remove("debug_output.txt")

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

    decay_steps = 5             # Number of steps before applying the exploration rate decay
    max_memory_size = 5000       # Maximum size of the experience replay memory

    if os.path.exists('/content/drive'):  # Google Colab environment
        file_path = google_colab_path
    else:  # Local environment
        file_path = local_path

    data_loader_easy = DataLoader("./resources/sudoku_mini_easy.csv", 8)

    easy_puzzles = data_loader_easy.get_puzzles()
    with strategy.scope():
        env = SudokuEnvironment(easy_puzzles, 10) # pass the list of puzzles to the environment
        agent = QLearningAgent(learning_rate, discount_factor, exploration_rate, exploration_decay, strategy, decay_steps, max_memory_size, file_path)
        trainer_easy = SudokuTrainer(agent, env, data_loader_easy)

        epochs = 10
        allowed_steps = 100
        batch_size = 20
        target_update_interval = 100

        trainer_easy.train(epochs, allowed_steps, batch_size, target_update_interval)

if __name__ == "__main__":
    __main__()