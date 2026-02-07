import numpy as np
import tensorflow as tf
from keras import Sequential, layers, optimizers, losses

CHAR_MAP = {
    '0': 0,  # Empty
    'W': 1,  # Wall
    'S': 2,  # Snake Body
    'G': 3,  # Green Apple
    'R': 4,  # Red Apple
    'H': 5   # Head (though head is not in the view usually)
}

class Agent:
    def __init__(self, board_size, learning_rate=0.001, gamma=0.95, debug=False):
        self.board_size = board_size
        self.gamma = gamma
        self.output_size = 4  # UP, DOWN, LEFT, RIGHT
        self.input_size = 4 * board_size
        self.optimizer = optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)  # Add gradient clipping
        self.loss_fn = losses.MeanSquaredError()
        self.debug = debug
        self.train_count = 0  # Counter for training steps

        # Build the Neural Network (with batch normalization for better learning)
        self.model = Sequential([
            layers.Input(shape=(self.input_size,)),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(32, activation='relu'),
            layers.Dense(self.output_size, activation='linear') 
            # Linear activation for the output because Q-values can be anything
        ])
        
        if self.debug:
            print(f"\n{'='*60}")
            print(f"🔧 Agent initialized:")
            print(f"  Board size: {board_size}")
            print(f"  Input size: {self.input_size} (4 directions × {board_size})")
            print(f"  Output size: {self.output_size}")
            print(f"  Learning rate: {learning_rate}")
            print(f"  Gamma: {gamma}")
            print(f"{'='*60}\n")

    def process_view_tf(self, view, board_size):
        """
        Converts the raw view lists into a NumPy array suitable for TensorFlow.
        Returns shape: (1, 4 * board_size)
        """
        input_vector = []
        
        for i, arm in enumerate(view):
            # Map chars to ints
            num_arm = [CHAR_MAP.get(c, 0) for c in arm]
            
            # Pad with 0s to reach board_size
            padding = [0] * (board_size - len(num_arm))
            
            # Extend the vector
            final_arm = num_arm + padding
            
            # Safety truncate
            input_vector.extend(final_arm[:board_size])
        
        # Convert to numpy and normalize to [0, 1] range
        result = np.array(input_vector, dtype=np.float32).reshape(1, -1)
        result = result / 5.0  # Max value in CHAR_MAP is 5 (H)
        
        # Debug first call
        if self.debug and self.train_count == 0:
            print(f"\n🔍 DEBUG process_view_tf (first call):")
            print(f"  Raw view: {view}")
            print(f"  View lengths: {[len(arm) for arm in view]}")
            print(f"  Processed input shape: {result.shape}")
            print(f"  Input sample (first 20 values): {result[0][:20]}")
            print(f"  Input min/max: {result.min():.2f} / {result.max():.2f}")
        
        return result

    def get_action(self, view, epsilon=0):
        """
        Decides the move based on Epsilon-Greedy strategy.
        """
        # Exploration
        if np.random.rand() < epsilon:
            action = np.random.randint(0, self.output_size)
            if self.debug and self.train_count < 3:
                print(f"  🎲 Exploration: random action {action}")
            return action
            
        # Exploitation (Prediction)
        state = self.process_view_tf(view, self.board_size)
        
        # training=False ensures layers like Dropout (if any) behave correctly
        pred_q_values = self.model(state, training=False) 
        
        # Debug first few predictions
        if self.debug and self.train_count < 3:
            print(f"\n🔍 DEBUG get_action:")
            print(f"  Q-values: {pred_q_values[0].numpy()}")
            print(f"  Max Q: {np.max(pred_q_values[0]):.3f}")
            print(f"  Selected action: {np.argmax(pred_q_values[0])}")
        
        # Argmax returns the index of the best move
        return np.argmax(pred_q_values[0])
    
    @tf.function # This decorator speeds up execution by compiling the graph
    def _train_step_compiled(self, state, action_idx, target_q):
        """
        Internal compiled function to update weights.
        Using GradientTape allows us to control exactly which gradients to update.
        """
        with tf.GradientTape() as tape:
            # 1. Predict Q-values for the current state
            q_values = self.model(state, training=True)
            
            # 2. Create a mask so we only calculate loss for the action taken
            # We want: Loss = (Q_target - Q_predicted[action])^2
            # But Keras loss functions expect shape matches.
            # So we create a "one_hot" mask to pick just the relevant Q-value.
            
            one_hot_action = tf.one_hot(action_idx, self.output_size)
            
            # We multiply by the mask to isolate the Q-value of the action taken
            current_q = tf.reduce_sum(q_values * one_hot_action, axis=1)
            
            # Calculate loss
            loss = self.loss_fn(target_q, current_q)
            
        # 3. Calculate Gradients and Update
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def train(self, current_view, action_idx, reward, next_view, done):
        """
        Orchestrates the Bellman Equation calculation and calls the training step.
        """
        self.train_count += 1
        
        # Convert inputs to Tensors/Arrays
        state = self.process_view_tf(current_view, self.board_size)
        next_state = self.process_view_tf(next_view, self.board_size)
        
        # 1. Calculate the Target Q-Value (Bellman Equation)
        # Q_new = r + gamma * max(Q(next_state))
        
        target_q = reward
        if not done:
            next_q_values = self.model(next_state, training=False)
            max_next_q = np.max(next_q_values[0])
            target_q = reward + self.gamma * max_next_q
        
        # Debug first few training steps
        if self.debug and self.train_count <= 3:
            print(f"\n🔍 DEBUG train (step {self.train_count}):")
            print(f"  Action: {action_idx} ({'UP' if action_idx==0 else 'DOWN' if action_idx==1 else 'LEFT' if action_idx==2 else 'RIGHT'})")
            print(f"  Reward: {reward}")
            print(f"  Done: {done}")
            if not done:
                print(f"  Next Q-values: {next_q_values[0].numpy()}")
                print(f"  Max next Q: {max_next_q:.3f}")
            print(f"  Target Q: {target_q:.3f}")
        
        # Convert target to tensor for the graph
        target_q_tensor = tf.convert_to_tensor([target_q], dtype=tf.float32)
        
        # 2. Run the optimization step
        loss = self._train_step_compiled(state, action_idx, target_q_tensor)
        
        # Debug loss
        if self.debug and self.train_count <= 3:
            print(f"  Loss: {loss.numpy():.6f}")
        
        # Print every 100 training steps
        if self.debug and self.train_count % 100 == 0:
            # Get current Q-values to see if network is learning
            current_q_values = self.model(state, training=False)
            print(f"\n📊 Training step {self.train_count}:")
            print(f"  Loss: {loss.numpy():.6f}")
            print(f"  Q-values: {current_q_values[0].numpy()}")
            print(f"  Q-value range: [{current_q_values[0].numpy().min():.2f}, {current_q_values[0].numpy().max():.2f}]")
        if not done:
            next_q_values = self.model(next_state, training=False)
            max_next_q = np.max(next_q_values[0])
            target_q = reward + self.gamma * max_next_q
        
        # Convert target to tensor for the graph
        target_q_tensor = tf.convert_to_tensor([target_q], dtype=tf.float32)
        
        # 2. Run the optimization step
        loss = self._train_step_compiled(state, action_idx, target_q_tensor)