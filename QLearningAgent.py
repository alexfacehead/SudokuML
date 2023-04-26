import tensorflow as tf
from keras.optimizers.schedules.learning_rate_schedule import ExponentialDecay
from typing import List, Tuple
from collections import deque
import re
import os

class QLearningAgent():
    def __init__(self, learning_rate: float, discount_factor: float, exploration_rate: \
                 float, exploration_decay: float, tpu_strategy: tf.distribute.TPUStrategy, decay_steps: \
                    int, max_memory_size: int, file_path: str):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.tpu_strategy = tpu_strategy
        self.memory = deque(maxlen=max_memory_size)
        self.file_path = file_path

        self.exploration_decay_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.exploration_rate,
            decay_steps=decay_steps,
            decay_rate=exploration_decay,
            staircase=True
        )
        
        with self.tpu_strategy.scope():
            self.model = self.create_q_network()
            self.target_model = self.create_q_network()
            self.update_target_q_network()

    # Generalizable method
    def _make_model(self, conv_layers: int, conv_filters: List[int], dense_layers: int, dense_units: List[int]) -> tf.keras.Model:
        inputs = tf.keras.Input(shape=(9, 9, 1))
        
        x = inputs
        for i in range(conv_layers):
            x = tf.keras.layers.Conv2D(conv_filters[i], kernel_size=3, padding='same', activation='relu')(x)
        
        x = tf.keras.layers.Flatten()(x)
        
        for i in range(dense_layers):
            x = tf.keras.layers.Dense(dense_units[i], activation='relu')(x)
        
        outputs = tf.keras.layers.Dense(9 * 9 * 9)(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='huber')
        
        return model

    # Specific method to make a Q-network architecture
    def create_q_network(self) -> tf.keras.Model:
        conv_layers = 2
        conv_filters = [64, 128]
        dense_layers = 2
        dense_units = [512, 9 * 9 * 9]
        
        return self.make_model(conv_layers, conv_filters, dense_layers, dense_units)
    
    # available_actions : List[Tuple[int, int, int]]
    # is train is True, we want exploration. if train is False, we want only exploitation/learned policy
    def choose_action(self, state: tf.Tensor, available_actions: List[Tuple[int, int, int]], train: bool=True) -> Tuple[int, int, int]:
        if train and tf.random.uniform(()) <= self.exploration_rate:
            return self.explore(available_actions)
        else:
            return self.exploit(state, available_actions)
    
    def explore(self, available_actions: List[Tuple[int, int, int]]) -> Tuple[int, int, int]:
        shuffled_actions = tf.random.shuffle(available_actions)
        return tuple(tf.gather(shuffled_actions, 0).numpy())

    def exploit(self, state: tf.Tensor, available_actions: List[Tuple[int, int, int]]) -> Tuple[int, int, int]:
        q_values = self.model.predict(tf.reshape(state, (1, 9, 9, 1)))
        q_values = tf.reshape(q_values, (9, 9, 9))
        mask = tf.zeros((9, 9, 9))
        for action in available_actions:
            mask = mask + tf.reshape(tf.one_hot(action, 9 * 9 * 9), (9, 9, 9)) # use tf.reshape instead of numpy
        masked_q_values = q_values * mask
        best_action = tf.unravel_index(tf.argmax(masked_q_values), masked_q_values.shape)
        return tuple(best_action.numpy())

    def replay(self, batch_size: int) -> None:
        batch = self.create_experience_batch(batch_size)

        states, target_q_values = [], []
        for state, action, reward, next_state, done in batch:
            if done:
                target_q_value = reward
            else:
                next_q_values = self.target_model.predict(tf.reshape(next_state, (1, 9, 9, 1)))
                target_q_value = reward + self.discount_factor * tf.reduce_max(next_q_values)

            current_q_values = self.model.predict(tf.reshape(state, (1, 9, 9, 1)))
            current_q_values = tf.reshape(current_q_values, (9, 9, 9))
            current_q_values = current_q_values.numpy()
            current_q_values[action] = target_q_value

            states.append(state)
            target_q_values.append(current_q_values)

        # create a dataset from lists of tensors directly
        dataset = tf.data.Dataset.from_tensor_slices((states, target_q_values))
        # optionally shuffle and batch the dataset
        dataset = dataset.shuffle(buffer_size=len(states)).batch(batch_size)

        with self.tpu_strategy.scope():
            # iterate over the dataset and train on each batch
            for states_batch, target_q_values_batch in dataset:
                self.model.train_on_batch(states_batch, target_q_values_batch)
    
    def remember(self, state: tf.Tensor, action: Tuple[int, int, int], reward: float, next_state: tf.Tensor, done: bool) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def create_experience_batch(self, batch_size: int) -> List[Tuple[tf.Tensor, Tuple[int, int, int], float, tf.Tensor, bool]]:
        batch = []
        if len(self.memory) < batch_size:
            batch_size = len(self.memory)

        indices = tf.random.shuffle(tf.range(len(self.memory)))[:batch_size]
        for idx in indices:
            batch.append(self.memory[idx.numpy()])

        return batch


    def update_target_q_network(self) -> None:
        self.target_model.set_weights(self.model.get_weights())

    def save_weights(self) -> None:
            next_epoch = self.get_highest_epoch() + 1
            new_file_name = f"epoch_{next_epoch}.pt"
            new_file_path = os.path.join(self.file_path, new_file_name)
            self.model.save_weights(new_file_path)

    def load_weights(self) -> None:
        highest_epoch = self.get_highest_epoch()
        if highest_epoch == -1:
            raise FileNotFoundError(f"No weights file found in {self.file_path} directory")
        else:
            highest_file_name = f"epoch_{highest_epoch}.pt"
            highest_file_path = os.path.join(self.file_path, highest_file_name)
            self.model.load_weights(highest_file_path)

    def get_highest_epoch(self) -> int:
        files = os.listdir(self.file_path)
        pattern = re.compile(r"epoch_(\d+)\.pt")
        files = [f for f in files if pattern.match(f)]
        if not files:
            return -1
        else:
            epochs = [int(pattern.match(f).group(1)) for f in files]
            return max(epochs)

    def set_exploration_rate(self, rate: float):
        self.exploration_rate = rate

    def decay_exploration_rate(self, step: int):
        self.exploration_rate = self.exploration_decay_schedule(step)