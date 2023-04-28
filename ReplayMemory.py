import tensorflow as tf
class ReplayMemory():
    def __init__(self, max_memory_size):
        # Initialize the class attributes with zeros and a fixed shape and dtype
        self.max_memory_size = max_memory_size
        self.state = tf.Variable(tf.zeros((max_memory_size, 9, 9, 1)), dtype=tf.float32)
        self.action = tf.Variable(tf.zeros((max_memory_size, 3)), dtype=tf.int32)
        self.reward = tf.Variable(tf.zeros((max_memory_size, 1)), dtype=tf.float32)
        self.next_state = tf.Variable(tf.zeros((max_memory_size, 9, 9, 1)), dtype=tf.float32)
        self.done = tf.Variable(tf.zeros((max_memory_size, 1)), dtype=tf.bool)
        self.count = 0 # Keep track of how many experiences are stored

    def add(self, state, action, reward, next_state, done):
        # Update the class attributes with new experiences at the current index
        index = self.count % self.max_memory_size
        self.state[index].assign(state)
        self.action[index].assign(action)
        self.reward[index].assign(reward)
        self.next_state[index].assign(next_state)
        self.done[index].assign(done)
        self.count += 1

    def sample(self, batch_size):
        # Create a dataset from the class attributes and sample a batch of experiences
        dataset = tf.data.Dataset.from_tensor_slices((self.state, self.action, self.reward, self.next_state, self.done))
        dataset = dataset.shuffle(buffer_size=self.max_memory_size).batch(batch_size)
        return next(iter(dataset))