import tensorflow as tf
import numpy as np
from keras.optimizers.schedules.learning_rate_schedule import ExponentialDecay
from typing import List, Tuple

class QLearningAgent():
    def __init__(self, learning_rate: float, discount_factor: float, exploration_rate: float, exploration_decay: float, tpu_strategy: tf.distribute.TPUStrategy, decay_steps: int, max_memory_size: int):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.tpu_strategy = tpu_strategy
        self.memory = deque(maxlen=max_memory_size)

        self.exploration_decay_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.exploration_rate,
            decay_steps=decay_steps,
            decay_rate=exploration_decay,
            staircase=True
        )
        
        with self.tpu_strategy.scope():
            self.model = self.create_q_network()

    def make_model(self, conv_layers: int, conv_filters: List[int], dense_layers: int, dense_units: List[int]) -> tf.keras.Model:
        inputs = tf.keras.Input(shape=(9, 9, 1))
        
        x = inputs
        for i in range(conv_layers):
            x = tf.keras.layers.Conv2D(conv_filters[i], kernel_size=3, padding='same', activation='relu')(x)
        
        x = tf.keras.layers.Flatten()(x)
        
        for i in range(dense_layers):
            x = tf.keras.layers.Dense(dense_units[i], activation='relu')(x)
        
        outputs = tf.keras.layers.Dense(9 * 9 * 9)(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        
        return model

    def create_q_network(self):
        conv_layers = 2
        conv_filters = [64, 128]
        dense_layers = 2
        dense_units = [512, 9 * 9 * 9]
        
        return self.make_model(conv_layers, conv_filters, dense_layers, dense_units)
    
    # available_actions : List[Tuple[int, int, int]]
    # is train is True, we want exploration. if train is False, we want only exploitation/learned policy
    def choose_action(self, state: np.ndarray, available_actions: List[Tuple[int, int, int]], train: bool = True) -> Tuple[int, int, int]:
        if train and np.random.rand() < self.exploration_rate:
            return self.explore(available_actions)
        else:
            return self.exploit(state, available_actions)
    
    def explore(self, available_actions: List[Tuple[int, int, int]]) -> Tuple[int, int, int]:
        return np.random.choice(available_actions)

    def exploit(self, state: np.ndarray, available_actions: List[Tuple[int, int, int]]) -> Tuple[int, int, int]:
        pass

    def remember(self, state: np.ndarray, action: Tuple[int, int, int], reward: float, next_state: np.ndarray, done: bool):
        self.memory.append((state, action, reward, next_state, done))

    def create_experience_batch(self, batch_size: int) -> List[Tuple[np.ndarray, Tuple[int, int, int], float, np.ndarray, bool]]:
        batch = []
        if len(self.memory) < batch_size:
            batch_size = len(self.memory)

        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        for idx in indices:
            batch.append(self.memory[idx])

        return batch

    def replay(self, batch_size: int):
        pass

    def update_target_q_network(self):
        pass

    def save_weights(self, file_path: str):
        self.model.save_weights(file_path)

    def load_weights(self, file_path: str):
        self.model.load_weights(file_path)

    def decay_exploration_rate(self, step: int):
        self.exploration_rate = self.exploration_decay_schedule(step)