import DataLoader
import QLearningAgent
import SudokuEnvironment
import tensorflow as tf
from typing import List
from QLearningAgent import print_debug_message

class SudokuTrainer():
    def __init__(self, agent: QLearningAgent, environment: SudokuEnvironment, data_loader: DataLoader):
        """A class that implements the training and evaluation logic for the sudoku solver.

        Attributes:
            agent: A QLearningAgent object that represents the agent that learns to solve sudoku puzzles.
            environment: A SudokuEnvironment object that represents the environment where the agent interacts.
            data_loader: A DataLoader object that provides the puzzles and solutions for training and evaluation.
            step: An integer that counts the number of steps taken by the agent.
            episode_rewards: A list of integers that stores the total rewards for each episode.
        """
        self.agent = agent
        self.environment = environment
        self.data_loader = data_loader
        self.episode_rewards = []
        self.total_steps = 0
        self.current_puzzle_steps = 0

    def train(self, epochs: int, allowed_steps: int, batch_size: int, target_update_interval: int) -> List[int]:
        solved_puzzles = 0  # Initialize solved puzzles counter
        for epoch in range(epochs):
            msg1 = "Epoch # " + str(epoch) + "\n"
            print(msg1)
            with open("debug_output.txt", "a") as f:
                f.write(msg1)

            # Evaluate after each epoch
            avg_reward = self.evaluate(20)
            print(f"Evaluation: Average reward over 20 episodes: {avg_reward}")
            QLearningAgent.print_debug_message(f"Evaluation: Average reward over 20 episodes: {avg_reward}")

            puzzles = self.data_loader.get_puzzles()
            puzzle_counter = 0
            for sudoku_board in puzzles:
                str1 = "Board: " + str(sudoku_board) + "\n"
                print(str1)
                QLearningAgent.print_debug_message(str1)
                puzzle_counter += 1
                msg2 = "Puzzle # " + str(puzzle_counter) + "\n"
                print(msg2)
                with open("debug_output.txt", "a") as f:
                    f.write(msg2)

                state = self.environment.reset(sudoku_board)
                self.current_puzzle_steps = 0
                episode_reward = 0

                for _ in range(allowed_steps):
                    self.agent.decay_exploration_rate(self.total_steps)
                    valid_actions = self.environment.get_valid_actions(state)
                    all_available_actions = self.environment.get_all_available_actions(state)
                    action = self.agent.choose_action(state, valid_actions, all_available_actions)
                    if action is None:
                        continue
                    next_state, reward, done = self.environment.step(action, valid_actions, all_available_actions)
                    self.agent.remember(state, action, reward, next_state, done)
                    episode_reward += reward
                    state = next_state
                    # new
                    self.environment.board = state
                    msg3 = "Running total step #" + str(self.total_steps) + "\n"
                    msg3 = msg3 + "Puzzle step # " + str(self.current_puzzle_steps) + "\n" + "Chosen action: " + str(QLearningAgent.format_action_tuple(action)) + "\n" + "Reward: " + str(reward) + "\n" + \
                    "Episode reward: " + str(episode_reward) + "\n"

                    if done or self.current_puzzle_steps == allowed_steps - 1:
                        break

                    self.agent.replay(batch_size)

                    if self.total_steps % target_update_interval == 0:
                        self.agent.update_target_q_network()

                    exploration_rate = self.agent.exploration_rate
                    if isinstance(exploration_rate, tf.Tensor):
                        exploration_rate = exploration_rate.numpy().item()
                    msg3 = msg3 + "\n" + "Exploration rate (epsilon): " + str(exploration_rate) + "\n"
                    print(msg3)
                    with open("debug_output.txt", "a") as f:
                        f.write(msg3)

                    self.total_steps += 1
                    self.current_puzzle_steps += 1

                self.episode_rewards.append(episode_reward)
                if done:
                    solved_puzzles += 1  # Increment solved puzzles counter

                # Print the number of solved puzzles for every 20 puzzles
                if puzzle_counter % 20 == 0:
                    print(f"Solved puzzles in the last 20: {solved_puzzles}")
                    QLearningAgent.print_debug_message(f"Solved puzzles in the last 20: {solved_puzzles}")
                    solved_puzzles = 0  # Reset the solved puzzles counter

            print("Saving weights for epoch " + str(epoch))
            self.agent.save_weights()
        return self.episode_rewards

    def evaluate(self, episodes: int) -> float:
        print_debug_message("Begin evaluation block!")
        episode_rewards = tf.Variable(tf.zeros([episodes], dtype=tf.float32))

        for i in range(episodes):
            #print_debug_message(f"Episode {i + 1} of {episodes} in evaluate - before loop")
            #print_debug_message("Getting a random board")
            sudoku_board = self.data_loader.get_random_board()
            #print_debug_message("Random board obtained")
            #print_debug_message("Resetting the environment")
            state = self.environment.reset()
            #print_debug_message("Environment reset")
            
            episode_reward = 0
            done = False

            while not done:
                #print_debug_message("First done: " + str(done))
                valid_actions = self.environment.get_valid_actions(state)
                all_available_actions = self.environment.get_all_available_actions(state)
                action = self.agent.choose_action(state, valid_actions, all_available_actions, train=False)
                next_state, reward, done = self.environment.step(action, valid_actions, all_available_actions)
                episode_reward += reward
                state = next_state

                #print_debug_message("Current state: " + str(state))
                #print_debug_message("Action: " + str(action))
                #print_debug_message("Reward: " + str(reward))

            #print_debug_message("Attempting assign!")
            episode_rewards[i].assign(episode_reward)
            #print_debug_message(f"Assigned episode reward {episode_reward} to index {i}")
            #print_debug_message(str(done))
            #print_debug_message(f"Episode {i + 1} of {episodes} in evaluate - after loop")

        print_debug_message("episode rewards: " + str(episode_rewards))
        return tf.reduce_mean(episode_rewards)




