import numpy as np
import tensorflow as tf
from keras import Sequential, layers, optimizers, losses
from collections import deque
import random

CHAR_MAP = {
    '0': 0,  # Empty
    'W': 1,  # Wall
    'S': 2,  # Snake Body
    'G': 3,  # Green Apple
    'R': 4,  # Red Apple
    'H': 5   # Head
}

class Agent:
    def __init__(self, board_size, learning_rate=0.0005, gamma=0.995, debug=False):
        self.board_size = board_size
        self.gamma = gamma
        self.output_size = 4  # UP, DOWN, LEFT, RIGHT
        # One-Hot: 4 directions * board_size cells * 6 possible categories
        self.num_categories = 6
        self.input_size = 4 * board_size * self.num_categories
        
        self.optimizer = optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
        self.loss_fn = losses.MeanSquaredError()
        self.debug = debug
        self.train_count = 0
        
        # Experience Replay Memory
        self.memory = deque(maxlen=50000)
        self.batch_size = 256
        self.target_update_frequency = 1000 # steps
        
        # Main Model (the one we train)
        self.model = self._build_model()
        
        # Target Model (the one we use for targets, periodic updates)
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = Sequential([
            layers.Input(shape=(self.input_size,)),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.1), # Slight dropout to prevent overfitting to specific board positions
            layers.Dense(128, activation='relu'),
            layers.Dense(self.output_size, activation='linear')
        ])
        return model

    def update_target_model(self):
        """Copies weights from main model to target model"""
        self.target_model.set_weights(self.model.get_weights())

    def process_view_tf(self, view, board_size):
        """
        Converts raw view into a One-Hot encoded vector.
        Each cell becomes a [0,0,0,0,0,0] vector with a 1 in the category index.
        """
        # Create a zero matrix [Total_Cells, 6_Categories]
        total_cells = 4 * board_size
        one_hot_matrix = np.zeros((total_cells, self.num_categories), dtype=np.float32)
        
        cell_idx = 0
        for arm in view:
            # Truncate arm to agent's board_size (Option 1)
            arm_truncated = list(arm[:board_size])
            
            # Virtual Wall logic: If no wall is seen in the 10-cell range, 
            # place a virtual wall at the last cell to simulate 10x10 boundaries.
            # This prevents the IA from getting lost in large empty spaces.
            if len(arm_truncated) == board_size and 'W' not in arm_truncated:
                arm_truncated[-1] = 'W'

            # Convert arm chars to indices
            for c in arm_truncated:
                if cell_idx < total_cells:
                    cat_idx = CHAR_MAP.get(str(c).strip()[-1], 0)
                    one_hot_matrix[cell_idx, cat_idx] = 1.0
                    cell_idx += 1
            
            # Padding for the arm if it's shorter than board_size (e.g. near walls)
            remaining_in_arm = board_size - (cell_idx % board_size)
            if remaining_in_arm < board_size:
                cell_idx += remaining_in_arm

        # Flatten to [1, 4 * board_size * 6]
        return one_hot_matrix.flatten().reshape(1, -1)

    def get_action(self, view, epsilon=0):
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.output_size)
            
        state = self.process_view_tf(view, self.board_size)
        pred_q_values = self.model(state, training=False)
        return np.argmax(pred_q_values[0])

    def remember(self, state_view, action, reward, next_state_view, done):
        """Store experiences in memory"""
        self.memory.append((state_view, action, reward, next_state_view, done))

    @tf.function
    def _train_step_compiled(self, states, actions, targets):
        with tf.GradientTape() as tape:
            all_q_values = self.model(states, training=True)
            masks = tf.one_hot(actions, self.output_size)
            current_qs = tf.reduce_sum(all_q_values * masks, axis=1)
            loss = self.loss_fn(targets, current_qs)
            
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def replay(self):
        """Train on a batch of experiences from memory"""
        if len(self.memory) < self.batch_size:
            return None
            
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = []
        next_states = []
        actions = []
        rewards = []
        dones = []
        
        for sv, a, r, nsv, d in minibatch:
            states.append(self.process_view_tf(sv, self.board_size)[0])
            next_states.append(self.process_view_tf(nsv, self.board_size)[0])
            actions.append(a)
            rewards.append(r)
            dones.append(d)
            
        states = np.array(states)
        next_states = np.array(next_states)
        actions = np.array(actions, dtype=np.int32)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        
        # Use target network for stability
        next_q_values = self.target_model(next_states, training=False)
        max_next_q = np.max(next_q_values, axis=1)
        
        targets = rewards + (1 - dones) * self.gamma * max_next_q
        
        loss = self._train_step_compiled(states, actions, targets)
        self.train_count += 1
        
        # Periodic update of target model
        if self.train_count % self.target_update_frequency == 0:
            self.update_target_model()
            if self.debug:
                print(f"🔄 Target model updated at step {self.train_count}")
                
        return loss.numpy()

    def train(self, current_view, action_idx, reward, next_view, done):
        pass