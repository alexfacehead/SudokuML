import DataLoader
import QLearningAgent
import SudokuEnvironment
import tensorflow as tf
print(tf.__version__)
print("Available devices:", tf.config.list_logical_devices())
from typing import List
from QLearningAgent import print_debug_msg
import logging

# Threading / popups
import tkinter as tk
from tkinter import messagebox
import threading

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

    def train(self, epochs: int, allowed_steps: int, batch_size: int, target_update_interval: int, force: bool, use_weights=False) -> List[int]:
        if use_weights:
            print("Attempting to load weights...")
            print_debug_msg("Attempting to load weights...")
            self.agent.load_weights()
        solved_puzzles = 0  # Initialize solved puzzles counter
        for epoch in range(epochs):
            msg1 = "Epoch # " + str(epoch) + "\n"
            print(msg1)
            with open("debug_output.txt", "a") as f:
                f.write(msg1)
            
            # Only by default evaluate AFTER the first epoch, or if the force flag is used
            if force or epoch > 0:
                num_tests = 10
                avg_reward, total_solved = self.evaluate(num_tests)
                total_msg_formatted = str(total_solved) + " / " + str(num_tests)
                print(total_msg_formatted)
                print(f"Evaluation: Average reward over {num_tests} episodes: {avg_reward}")
                print_debug_msg(f"Evaluation: Average reward over 20 episodes: {avg_reward}")
                print_debug_msg(total_msg_formatted)
                #print("Saving weights")
                #print_debug_msg("Saving weights")
                #self.agent.save_weights()
                if force:
                    return avg_reward
            
            puzzles = self.data_loader.get_puzzles()
            puzzle_counter = 0
            for sudoku_board in puzzles:
                #checkpoint_number = 1000
                #if puzzle_counter % checkpoint_number == 0 and puzzle_counter != 0:
                #    threading.Thread(target=show_popup, args=(str(checkpoint_number) + " puzzles completed",)).start()
                puzzle_counter += 1
                msg2 = "Puzzle # " + str(puzzle_counter) + "\n"
                print(msg2)
                with open("debug_output.txt", "a") as f:
                    f.write(msg2)

                state = self.environment.reset(sudoku_board)
                self.current_puzzle_steps = 0
                episode_reward = 0

                for _ in range(allowed_steps):
                    is_solved = self.environment.is_solved()
                    if is_solved:
                        solved_puzzles += 1
                    self.agent.decay_exploration_rate(self.total_steps)
                    valid_actions = self.environment.get_valid_actions(state)
                    all_available_actions = self.environment.get_all_available_actions(state)
                    if not all_available_actions: # originally not valid_actions
                        if not self.environment.is_solved():
                            print_debug_msg("Failed.")
                        print_debug_msg("FINAL STATE: " + str(self.environment.board))
                        break
                    action = self.agent.choose_action(state, valid_actions, all_available_actions)
                    if action is None or not all_available_actions:
                        print_debug_msg("Got action: None or no available actions to take.")
                        break
                    next_state, reward, done = self.environment.step(action, valid_actions, all_available_actions)
                    if done:
                        print_debug_msg("Done hit")
                        break
                    print_debug_msg("Attempting to store memory...")
                    self.agent.remember(state, action, reward, next_state, done)
                    episode_reward += reward
                    state = next_state

                    msg3 = "Running total step #" + str(self.total_steps) + "\n"
                    msg3 = msg3 + "Puzzle step # " + str(self.current_puzzle_steps) + "\n" + "Chosen action: " + str(QLearningAgent.format_action_tuple(action)) + "\n" + "Reward: " + str(reward) + "\n" + \
                    "Episode reward: " + str(episode_reward) + "\n"
                    print_debug_msg(msg3)
                    if done or self.current_puzzle_steps == allowed_steps - 1:
                        print("hit done/self.current_puzzle_steps max block train")
                        break

                    self.agent.replay(batch_size)

                    if self.total_steps % target_update_interval == 0:
                        self.agent.update_target_q_network()

                    exploration_rate = self.agent.exploration_rate
                    if isinstance(exploration_rate, tf.Tensor):
                        exploration_rate = exploration_rate.numpy().item()
                    msg3 = "\n" + "Exploration rate (epsilon): " + str(exploration_rate) + "\n"
                    #print(msg3)
                    print_debug_msg(msg3)

                    self.total_steps += 1
                    self.current_puzzle_steps += 1

                self.episode_rewards.append(episode_reward)

                # Print the number of solved puzzles for every 20 puzzles
                if puzzle_counter % 20 == 0:
                    print(f"Solved puzzles in the last 20: {solved_puzzles}")
                    print_debug_msg(f"Solved puzzles in the last 20: {solved_puzzles}")
                    solved_puzzles = 0  # Reset the solved puzzles counter

            print("Saving weights for epoch " + str(epoch))
            self.agent.save_weights()
        return self.episode_rewards

    def evaluate(self, episodes: int) -> float:
        print_debug_msg("Begin evaluation block!")
        episode_rewards = tf.Variable(tf.zeros([episodes], dtype=tf.float32))
        total_solved = 0
        for i in range(episodes):
            sudoku_board = self.data_loader.get_random_board()
            state = self.environment.reset(sudoku_board)
            
            episode_reward = 0
            done = False

            while not done:
                valid_actions = self.environment.get_valid_actions(state)
                all_available_actions = self.environment.get_all_available_actions(state)
                action = self.agent.choose_action(state, valid_actions, all_available_actions, train=False)
                is_solved = self.environment.is_solved()

                if action is None or not all_available_actions or is_solved: # used to have not valid_actions
                    print_debug_msg("Is solved?:" + str(is_solved))
                    if is_solved:
                        print("End solved.")
                        total_solved += 1
                    break
                action_tuple = QLearningAgent.format_action_tuple(action)
                print_debug_msg("Chosen action: " + str(action_tuple))
                next_state, reward, done = self.environment.step(action, valid_actions, all_available_actions)
                episode_reward += reward
                state = next_state

            episode_rewards = tf.tensor_scatter_nd_update(episode_rewards, [[i]], [episode_reward])

        print_debug_msg("Evaluation results:\nEpisode rewards: " + str(episode_rewards))
        return tf.reduce_mean(episode_rewards), total_solved
    
def show_popup(message: str):
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo("Threshold Reached", message)
    root.destroy()