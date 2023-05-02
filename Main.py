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
import itertools
from QLearningAgent import print_debug_msg

load_dotenv()
google_colab_path = os.getenv("google_colab_path")
local_path = os.getenv("local_path")
if local_path is None:
    local_path = os.getcwd()

def __main__():
    """Run the main program.

    Args:
        None

    Returns:
        None
    """
    hyperparameter_space = {
        'learning_rate': [0.01, 0.1, 0.2],
        'discount_factor': [0.9, 0.99, 0.999],
        'exploration_rate': [0.5, 1.0],
        'exploration_decay': [0.95, 0.99, 0.995],
        'allowed_steps': [50, 100, 200],
        'batch_size': [10, 20, 40],
        'target_update_interval': [50, 100, 200],
        'number_of_puzzles': [5, 10, 20],
        'decay_steps': [500, 1000, 2000],
        'max_memory_size': [5000, 10000, 20000]
    }

    # Check if we're using colab
    if os.path.exists('/content/drive'):  # Google Colab environment
        file_path = google_colab_path
    else:  # Local environment
        file_path = local_path

    # Argparse stuff
    parser = argparse.ArgumentParser(description="Sudoku reinforcement learning")
    parser.add_argument("--fresh", action="store_true", help="Delete debug_output.txt if it exists")
    parser.add_argument("--force", action="store_true", help="Use --force to enable the force flag")
    args = parser.parse_args()

    # Set force variable to True if --force flag is used, otherwise set it to False
    force = args.force

    if args.fresh and os.path.exists("debug_output.txt"):
        os.remove("debug_output.txt")

    learning_rate = 0.1          # The learning rate for the Q-Learning algorithm
    discount_factor = 0.99       # The discount factor for future rewards
    exploration_rate = 0.5       # Initial exploration rate (epsilon) for the epsilon-greedy strategy
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

    # Set the relatively fixed hyper parameters
    decay_steps = 5             # Number of steps before applying the exploration rate decay
    max_memory_size = 5000       # Maximum size of the experience replay memory

    # Change if you want different puzzles!
    data_loader_easy = DataLoader("./resources/sudoku_mini_easy.csv", 8)
    data_loader_easy_2 = DataLoader("./resources/sudoku_mini_small.csv", 12)
    data_loader_easy_3 = DataLoader("./resources/sudoku_mini_small.csv", 16)
    data_loader_easy_4 = DataLoader("./resources/sudoku_mini_small.csv", 20)
    data_loader_medium = DataLoader("./resources/sudoku_mini_medium.csv", 24)

    easy_puzzles = data_loader_easy.get_puzzles()
    with strategy.scope():
        env = SudokuEnvironment(easy_puzzles, max_incorrect_moves=20) # pass the list of puzzles to the environment
        agent = QLearningAgent(learning_rate, discount_factor, exploration_rate, exploration_decay, strategy, decay_steps, max_memory_size, file_path)
        trainer_easy = SudokuTrainer(agent, env, data_loader_easy_2)

        epochs = 10
        allowed_steps = 100
        batch_size = 20
        target_update_interval = 100
        

        #best_hyperparameters, best_performance = grid_search(hyperparameter_space, strategy, data_loader_easy, decay_steps, max_memory_size, file_path, force)
        #print("Grid search results: Best hyperparameters found: ", best_hyperparameters)
        #print("Grid search results: Best performance: ", best_performance)
        #print_debug_msg("Grid search results: Best hyperparameters found: {}".format(best_hyperparameters))
        #print_debug_msg("Grid search results: Best performance: {}".format(best_performance))
        # Standard training loop with predefined params
        performance = trainer_easy.train(epochs, allowed_steps, batch_size, target_update_interval, force)

def train_and_tune(hyperparameters, strategy, data_loader, decay_steps, max_memory_size, file_path, force):
    easy_puzzles = data_loader.get_puzzles()[:hyperparameters['number_of_puzzles']]
    with strategy.scope():
        env = SudokuEnvironment(easy_puzzles, max_incorrect_moves=20)
        agent = QLearningAgent(hyperparameters['learning_rate'], hyperparameters['discount_factor'], hyperparameters['exploration_rate'], hyperparameters['exploration_decay'], strategy, decay_steps, max_memory_size, file_path)
        trainer_easy = SudokuTrainer(agent, env, data_loader)

        # You may need to modify the trainer to return the performance metric
        episode_rewards = trainer_easy.train(1, hyperparameters['allowed_steps'], hyperparameters['batch_size'], hyperparameters['target_update_interval'], force)

    # Calculate the average performance
    avg_performance = sum(episode_rewards) / len(episode_rewards)

    return avg_performance

def grid_search(hyperparameter_space, strategy, data_loader, decay_steps, max_memory_size, file_path, force):
    keys, values = zip(*hyperparameter_space.items())
    best_performance = -float('inf')
    best_hyperparameters = None

    for combination in itertools.product(*values):
        hyperparameters = dict(zip(keys, combination))
        # Pass the additional arguments: strategy, data_loader, decay_steps, max_memory_size, and file_path
        performance = train_and_tune(hyperparameters, strategy, data_loader, decay_steps, max_memory_size, file_path, force)

        if performance > best_performance:
            best_performance = performance
            best_hyperparameters = hyperparameters

    return best_hyperparameters, best_performance

if __name__ == "__main__":
    __main__()