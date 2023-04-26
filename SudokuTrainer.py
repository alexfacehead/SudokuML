import DataLoader
import QLearningAgent
import SudokuEnvironment
import tensorflow as tf
from typing import List

class SudokuTrainer():
    def __init__(self, agent: QLearningAgent, environment: SudokuEnvironment, data_loader: DataLoader):
        self.agent = agent
        self.environment = environment
        self.data_loader = data_loader
        self.step = 0
        self.episode_rewards = []

    def train(self, epochs: int, allowed_steps: int, batch_size: int, target_update_interval: int) -> List[int]:
        for epoch in range(epochs):
            puzzles = self.data_loader.get_puzzles()
            for sudoku_board in puzzles:
                state = self.environment.reset(sudoku_board)
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
        episode_rewards = tf.Variable(tf.zeros([episodes], dtype=tf.float32))

        for i in range(episodes):
            sudoku_board = self.data_loader.get_random_board()
            state = self.environment.reset(sudoku_board)
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

    def play(self, sudoku_board: tf.Tensor) -> tf.Tensor:
        state = self.environment.reset(sudoku_board)
        done = False

        while not done:
            available_actions = self.environment.get_available_actions(state)
            if not available_actions:
                break
            action = self.agent.choose_action(state, available_actions, train=False)
            next_state, _, done = self.environment.step(action)
            state = next_state

        return state