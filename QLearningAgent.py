import tensorflow as tf
from keras.optimizers.schedules.learning_rate_schedule import ExponentialDecay
from typing import List, Tuple
from collections import deque
import re
import os
debug = False

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
            index = action[0] * 9 * 9 + action[1] * 9 + (action[2] - 1)  # Convert the 3D index to 1D index
            mask = mask + tf.reshape(tf.one_hot(index, 9 * 9 * 9), (9, 9, 9))
        masked_q_values = q_values * mask
        best_action = tf.unravel_index(tf.argmax(masked_q_values), masked_q_values.shape)
        return tuple(best_action)

    def replay(self, batch_size: int) -> None:
        msg1 = "Batch size: " + str(batch_size)
        if debug:
            print(msg1)
        else:
            with open("debug_output.txt", "a") as f:
                f.write(msg1 + "\n")

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.create_experience_batch(batch_size)
        msg2 = "state_batch shape: " + str(state_batch.shape) + "\n" + \
            "action_batch shape: " + str(action_batch.shape) + "\n" + \
            "reward_batch shape: " + str(reward_batch.shape) + "\n" + \
            "next_state_batch shape: " + str(next_state_batch.shape) + "\n" + \
            "done_batch shape: " + str(done_batch.shape)
        if debug:
            print(msg2)
        else:
            with open("debug_output.txt", "a") as f:
                f.write(msg2 + "\n")

        next_q_values = self.target_model.predict(next_state_batch)
        next_q_values = tf.reshape(next_q_values, (-1, 9, 9, 9))
        msg3 = "next_q_values shape: " + str(next_q_values.shape) + "\n"

        current_q_values = self.model.predict(state_batch)
        current_q_values = tf.reshape(current_q_values, (-1, 9, 9, 9))
        msg3 = msg3 + "current_q_values shape: " + str(current_q_values.shape) + "\n"

        value_indices = tf.cast(action_batch[:, 2], tf.int32) - 1
        gather_indices = tf.concat([tf.cast(action_batch[:, :2], tf.int32), tf.expand_dims(value_indices, axis=-1)], axis=-1)
        next_q_values_selected = tf.gather_nd(next_q_values, gather_indices, batch_dims=1)
        msg3 = msg3 + "next_q_values_selected shape: " + str(next_q_values_selected.shape) + "\n"

        next_q_values_selected = tf.expand_dims(next_q_values_selected, axis=-1)

        target_q_values = reward_batch + self.discount_factor * next_q_values_selected * (1 - done_batch)
        msg3 = msg3 + "target_q_values shape (before squeezing): " + str(target_q_values.shape) + "\n"

        target_q_values = tf.squeeze(target_q_values, axis=-1)
        msg3 = msg3 + "target_q_values shape (after squeezing): " + str(target_q_values.shape) + "\n"

        target_q_values = tf.expand_dims(target_q_values, axis=-1)
        target_q_values = tf.expand_dims(target_q_values, axis=-1)
        target_q_values = tf.expand_dims(target_q_values, axis=-1)
        msg3 = msg3 + "target_q_values shape (after expanding dimensions): " + str(target_q_values.shape) + "\n"

        mask = tf.one_hot(tf.cast(action_batch[:, 2], tf.int32), 9)
        mask = tf.expand_dims(mask, axis=1)
        mask = tf.expand_dims(mask, axis=1)
        mask = tf.broadcast_to(mask, tf.shape(current_q_values))
        msg3 = msg3 + "mask shape: " + str(mask.shape) + "\n"

        target_q_values = current_q_values * (1 - mask) + target_q_values * mask
        msg3 = msg3 + "target_q_values shape (after applying mask): " + str(target_q_values.shape) + "\n"

        states = state_batch
        target_q_values = tf.reshape(target_q_values, (-1, 9 * 9 * 9))
        msg3 = msg3 + "target_q_values shape (after reshaping): " + str(target_q_values.shape) + "\n"
        if debug:
            print(msg3)
        else:
            with open("debug_output.txt", "a") as f:
                f.write(msg3)

        dataset = tf.data.Dataset.from_tensor_slices((states, target_q_values))
        dataset = dataset.shuffle(buffer_size=tf.cast(tf.size(states), tf.int64)).batch(batch_size)

        with self.tpu_strategy.scope():
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

        msg = """Remembering experience:
        State: {}
        Action: {}
        Reward: {}
        Next state: {}
        Done: {}
        Memory size: {}
        """.format(state, format_action_tuple(action), reward, next_state, done, len(self.memory))
        if debug:
            print("\n" + str(msg))
        else:
            with open("debug_output.txt", "a") as f:
                f.write("\n" + str(msg))

    def create_experience_batch(self, batch_size: int) -> tf.Tensor:
        # create empty tensors of appropriate shapes for each element in the experience tuple
        state_batch = tf.zeros((0, 9, 9, 1))
        action_batch = tf.zeros((0, 3), dtype=tf.int32)
        reward_batch = tf.zeros((0, 1), dtype=tf.float32)
        next_state_batch = tf.zeros((0, 9, 9, 1))
        done_batch = tf.zeros((0, 1))

        if len(self.memory) < batch_size:
            batch_size = len(self.memory)

        indices = tf.random.shuffle(tf.range(len(self.memory)))[:batch_size]
        for idx in indices:
            # get the experience tuple from the memory
            experience = self.memory[idx.numpy()]
            # cast the data to match the tensor type
            experience = [tf.cast(x, dtype=t.dtype) for x, t in zip(experience, (state_batch, action_batch, reward_batch, next_state_batch, done_batch))]
            # reshape the tensors and concatenate them along the appropriate axes
            state_batch = tf.concat([state_batch, tf.reshape(experience[0], (1, 9, 9, 1))], axis=0)
            action_batch = tf.concat([action_batch, tf.reshape(experience[1], (1, 3))], axis=0)
            reward_batch = tf.concat([reward_batch, tf.reshape(experience[2], (1, 1))], axis=0)
            next_state_batch = tf.concat([next_state_batch, tf.reshape(experience[3], (1, 9, 9, 1))], axis=0)
            done_batch = tf.concat([done_batch, tf.reshape(experience[4], (1, 1))], axis=0)

        # return the batch as a tuple of tensors
        #print("state_batch shape:", state_batch.shape, "dtype:", state_batch.dtype)
        #print("action_batch shape:", action_batch.shape, "dtype:", action_batch.dtype)
        #print("reward_batch shape:", reward_batch.shape, "dtype:", reward_batch.dtype)
        #print("next_state_batch shape:", next_state_batch.shape, "dtype:", next_state_batch.dtype)
        #print("done_batch shape:", done_batch.shape, "dtype:", done_batch.dtype)

        # cast both the action batch and the done batch to float tensors
        action_batch = tf.cast(action_batch, dtype=tf.float32)
        done_batch = tf.cast(done_batch, dtype=tf.float32)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

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

        # Check if running on Google Colab
        on_colab = 'COLAB_GPU' in os.environ

        if on_colab:
            # Use TensorFlow save_weights method with ".ckpt" extension
            new_file_name = f"epoch_{next_epoch}.ckpt"
        else:
            # Use Keras save_weights method with ".pt" extension
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

def format_action_tuple(action_tuple : Tuple[int, int, int]) -> str:
        return f"({int(action_tuple[0])}, {int(action_tuple[1])}, {int(action_tuple[2])})"