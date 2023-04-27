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
        """Initialize the Q-learning agent with the parameters and models.

        Args:
            learning_rate: A float indicating the learning rate for the Q-network optimizer.
            discount_factor: A float indicating the discount factor for future rewards.
            exploration_rate: A float indicating the initial exploration rate (epsilon) for the epsilon-greedy strategy.
            exploration_decay: A float indicating the exploration rate decay factor.
            tpu_strategy: A tf.distribute.TPUStrategy object for distributed training on TPUs.
            decay_steps: An integer indicating the number of steps before applying the exploration rate decay.
            max_memory_size: An integer indicating the maximum size of the experience replay memory.
            file_path: A string indicating the file path for saving and loading the Q-network model.

        Returns:
            None
        """
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
        """Make a generic model with convolutional and dense layers.

        Args:
            conv_layers: An integer indicating the number of convolutional layers.
            conv_filters: A list of integers indicating the number of filters for each convolutional layer.
            dense_layers: An integer indicating the number of dense layers.
            dense_units: A list of integers indicating the number of units for each dense layer.

        Returns:
            A tf.keras.Model object representing the model.
        """
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
        """Create a Q-network architecture with specific parameters.

        Args:
            None

        Returns:
            A tf.keras.Model object representing the Q-network model.
        """
        conv_layers = 2
        conv_filters = [64, 128]
        dense_layers = 2
        dense_units = [512, 9 * 9 * 9]
        
        return self._make_model(conv_layers, conv_filters, dense_layers, dense_units)
    
    # available_actions : List[Tuple[int, int, int]]
    # is train is True, we want exploration. if train is False, we want only exploitation/learned policy
    def choose_action(self, state: tf.Tensor, available_actions: List[Tuple[int, int, int]], train: bool=True) -> Tuple[int, int, int]:
        """Choose an action based on the state and the available actions.

        Args:
            state: A tensor of shape (9, 9, 1) representing the current board state.
            available_actions: A list of tuples of the form (row, col, num) representing the possible actions.
            train: A boolean indicating whether to use exploration or exploitation.

        Returns:
            A tuple of the form (row, col, num) representing the chosen action.
        """
        if train and tf.random.uniform(()) <= self.exploration_rate:
            return self.explore(available_actions)
        else:
            return self.exploit(state, available_actions)
    
    def explore(self, available_actions: List[Tuple[int, int, int]]) -> Tuple[int, int, int]:
        """Choose an action randomly from the available actions.

        Args:
            available_actions: A list of tuples of the form (row, col, num) representing the possible actions.

        Returns:
            A tuple of the form (row, col, num) representing the chosen action.
        """
        shuffled_actions = tf.random.shuffle(available_actions)
        return tuple(tf.gather(shuffled_actions, 0))

    def exploit(self, state: tf.Tensor, available_actions: List[Tuple[int, int, int]]) -> Tuple[int, int, int]:
        """Choose an action based on the Q-values from the Q-network.

        Args:
            state: A tensor of shape (9, 9, 1) representing the current board state.
            available_actions: A list of tuples of the form (row, col, num) representing the possible actions.

        Returns:
            A tuple of the form (row, col, num) representing the chosen action.
        """
        q_values = self.model.predict(tf.reshape(state, (1, 9, 9, 1)))
        q_values = tf.reshape(q_values, (9, 9, 9))
        mask = tf.zeros((9, 9, 9))
        for action in available_actions:
            mask = mask + tf.reshape(tf.one_hot(action, 9 * 9 * 9), (9, 9, 9)) # use tf.reshape
        masked_q_values = q_values * mask
        best_action = tf.unravel_index(tf.argmax(masked_q_values), masked_q_values.shape)
        return tuple(best_action)

    def replay(self, batch_size: int) -> None:
        """Replay a batch of experiences and update the Q-network.

        Args:
            batch_size: An integer indicating the size of the batch.

        Returns:
            None
        """
        # get the batch tensor of shape (batch_size, 5)
        batch = self.create_experience_batch(batch_size)
        # unpack the batch tensor into five tensors of shape (batch_size, ...)
        state, action, reward, next_state, done = tf.unstack(batch, axis=1)

        # create empty tensors for states and target_q_values
        states = tf.zeros((0, 9, 9, 1))
        target_q_values = tf.zeros((0, 9, 9, 9))
        
        # iterate over the batch size dimension
        for i in range(batch_size):
            if done[i]:
                target_q_value = reward[i]
            else:
                next_q_values = self.target_model.predict(tf.reshape(next_state[i], (1, 9, 9, 1)))
                target_q_value = reward[i] + self.discount_factor * tf.reduce_max(next_q_values)

            current_q_values = self.model.predict(tf.reshape(state[i], (1, 9, 9, 1)))
            current_q_values = tf.reshape(current_q_values, (9, 9, 9))
            current_q_values = tf.tensor_scatter_nd_update(current_q_values, [action[i]], [target_q_value])

            # concatenate the state and current_q_values tensors to the existing ones
            states = tf.concat([states, tf.reshape(state[i], (1, 9, 9, 1))], axis=0)
            target_q_values = tf.concat([target_q_values, tf.reshape(current_q_values[i], (1, 9, 9, 9))], axis=0)

        # create a dataset from tensors directly
        dataset = tf.data.Dataset.from_tensor_slices((states, target_q_values))
        # optionally shuffle and batch the dataset
        dataset = dataset.shuffle(buffer_size=tf.size(states)).batch(batch_size)

        with self.tpu_strategy.scope():
            # iterate over the dataset and train on each batch
            for states_batch, target_q_values_batch in dataset:
                self.model.train_on_batch(states_batch, target_q_values_batch)
    
    def remember(self, state: tf.Tensor, action: Tuple[int, int, int], reward: float, next_state: tf.Tensor, done: bool) -> None:
        """Store an experience tuple in the replay memory.

        Args:
            state: A tensor of shape (9, 9, 1) representing the current board state.
            action: A tuple of the form (row, col, num) representing the action taken.
            reward: A float representing the reward received.
            next_state: A tensor of shape (9, 9, 1) representing the next board state.
            done: A boolean indicating whether the episode is over or not.

        Returns:
            None
        """
        self.memory.append((state, action, reward, next_state, done))

    def create_experience_batch(self, batch_size: int) -> tf.Tensor:
        """Create a batch of experiences from the replay memory.

        Args:
            batch_size: An integer indicating the size of the batch.

        Returns:
            A tensor of shape (batch_size, 5) representing a batch of experiences.
        """
        # create an empty tensor of shape (0, 5)
        batch = tf.zeros((0, 5))
        if len(self.memory) < batch_size:
            batch_size = len(self.memory)

        indices = tf.random.shuffle(tf.range(len(self.memory)))[:batch_size]
        for idx in indices:
            # get the experience tuple from the memory
            experience = self.memory[idx.numpy()]
            # convert the tuple to a tensor of shape (1, 5)
            experience_tensor = tf.stack(experience, axis=0)[tf.newaxis, ...]
            # concatenate the experience tensor to the batch tensor
            batch = tf.concat([batch, experience_tensor], axis=0)

        return batch



    def update_target_q_network(self) -> None:
        """Update the target Q-network with the weights from the Q-network.

        Args:
            None

        Returns:
            None
        """
        self.target_model.set_weights(self.model.get_weights())

    def save_weights(self) -> None:
        """Save the weights of the Q-network to a file.

        Args:
            None

        Returns:
            None
        """
        next_epoch = self.get_highest_epoch() + 1
        new_file_name = f"epoch_{next_epoch}.pt"
        new_file_path = os.path.join(self.file_path, new_file_name)
        self.model.save_weights(new_file_path)

    def load_weights(self) -> None:
        """Load the weights of the Q-network from a file.

        Args:
            None

        Returns:
            None
        """
        highest_epoch = self.get_highest_epoch()
        if highest_epoch == -1:
            raise FileNotFoundError(f"No weights file found in {self.file_path} directory")
        else:
            highest_file_name = f"epoch_{highest_epoch}.pt"
            highest_file_path = os.path.join(self.file_path, highest_file_name)
            self.model.load_weights(highest_file_path)

    def get_highest_epoch(self) -> int:
        """Get the highest epoch number from the file names in the file path.

        Args:
            None

        Returns:
            An integer representing the highest epoch number or -1 if no files are found.
        """
        files = os.listdir(self.file_path)
        pattern = re.compile(r"epoch_(\d+)\.pt")
        files = [f for f in files if pattern.match(f)]
        if not files:
            return -1
        else:
            epochs = [int(pattern.match(f).group(1)) for f in files]
            return max(epochs)

    def set_exploration_rate(self, rate: float):
        """Set the exploration rate to a given value.

        Args:
            rate: A float indicating the new exploration rate.

        Returns:
            None
        """
        self.exploration_rate = rate

    def decay_exploration_rate(self, step: int):
        """Decay the exploration rate according to the exponential decay schedule.

        Args:
            step: An integer indicating the current step.

        Returns:
            None
        """
        self.exploration_rate = self.exploration_decay_schedule(step)