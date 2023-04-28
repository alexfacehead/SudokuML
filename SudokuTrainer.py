import DataLoader
import QLearningAgent
import SudokuEnvironment
import tensorflow as tf
from typing import List

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
        self.step = 0
        self.episode_rewards = []

    def train(self, epochs: int, allowed_steps: int, batch_size: int, target_update_interval: int) -> List[int]:
        """Train the agent for a given number of epochs.

        Args:
            epochs: An integer indicating the number of epochs to train for.
            allowed_steps: An integer indicating the maximum number of steps allowed in each episode.
            batch_size: An integer indicating the size of the batch for replaying experiences.
            target_update_interval: An integer indicating the number of steps before updating the target Q-network.

        Returns:
            A list of integers representing the total rewards for each episode.
        """
        for epoch in range(epochs):
            puzzles = self.data_loader.get_puzzles()
            for sudoku_board in puzzles:
                state = self.environment.reset(sudoku_board)  # Pass the sudoku_board to the reset function
                self.step = 0
                episode_reward = 0

                for _ in range(allowed_steps):
                    available_actions = self.environment.get_available_actions(state)
                    action = self.agent.choose_action(state, available_actions)
                    next_state, reward, done = self.environment.step(action)
                    self.agent.remember(state, action, reward, next_state, done)
                    episode_reward += reward
                    state = next_state

                    if done or self.step == allowed_steps - 1:
                        break

                    self.agent.replay(batch_size)

                    if self.step % target_update_interval == 0:
                        self.agent.update_target_q_network()

                    self.agent.decay_exploration_rate(self.step)

                    self.step += 1

                self.episode_rewards.append(episode_reward)
                print("Saving weights for epoch " + str(epoch))
                self.agent.save_weights()

        return self.episode_rewards

    def evaluate(self, episodes: int) -> float:
        """Evaluate the agent on a given number of episodes.

        Args:
            episodes: An integer indicating the number of episodes to evaluate on.

        Returns:
            A float representing the average reward per episode.
        """
        episode_rewards = tf.Variable(tf.zeros([episodes], dtype=tf.float32))

        for i in range(episodes):
            sudoku_board = self.data_loader.get_random_board()
            state = self.environment.reset()
            episode_reward = 0
            done = False

            while not done:
                available_actions = self.environment.get_available_actions(state)
                action = self.agent.choose_action(state, available_actions, train=False)
                next_state, reward, done = self.environment.step(action)
                episode_reward += reward
                state = next_state

            episode_rewards[i].assign(episode_reward)

        return tf.reduce_mean(episode_rewards)