"""
Agent module.

Implements the Deep Q-Network (DQN) agent with experience replay and a
target network. The agent takes the snake's cross-shaped vision as input
and outputs Q-values for the four possible actions
(UP=0, DOWN=1, LEFT=2, RIGHT=3).

Two architectures are supported:
  - 'single' mode: one-hot encoding of cells, sized to a fixed board_size.
    Best 10x10 performance, but locked to a single board size.
  - 'multi'  mode: 16-D size-invariant feature vector (4 directions x
    [wall, body, green, red] inverse-distance). The same model can play any
    board size — used for the size-portability bonus.
"""

import numpy as np
import tensorflow as tf
from keras import Sequential, layers, optimizers, losses
from collections import deque
import random

# Mapping from board characters to one-hot category indices ('single' mode)
CHAR_MAP = {
    '0': 0,  # Empty cell
    'W': 1,  # Wall
    'S': 2,  # Snake body
    'G': 3,  # Green apple
    'R': 4,  # Red apple
    'H': 5,  # Snake head
}


class Agent:
    """
    DQN agent with experience replay and a periodically updated target network.

    Input depends on the configured mode:
      - 'single': one-hot vector of shape (4 * board_size * 6,)
      - 'multi' : size-invariant feature vector of shape (16,) — 4 directions
                  with [wall_dist, body_dist, green_dist, red_dist], each
                  encoded as 1/d (0 if not visible).
    """

    def __init__(self, board_size=None, mode='single',
                 learning_rate=0.000216, gamma=0.9646, debug=False,
                 batch_size=512, target_update_frequency=1000,
                 memory_size=150000,
                 hidden_layers=None, dropout_rate=None):
        if mode not in ('single', 'multi'):
            raise ValueError(
                f"Unknown mode '{mode}'. Use 'single' or 'multi'."
            )
        if mode == 'single' and board_size is None:
            raise ValueError("Single mode requires a board_size.")

        self.mode = mode
        self.board_size = board_size
        self.gamma = gamma
        self.output_size = 4  # UP, DOWN, LEFT, RIGHT
        self.num_categories = 6

        if mode == 'single':
            # 4 directions * board_size cells * 6 one-hot categories
            self.input_size = 4 * board_size * self.num_categories
        else:
            # 4 directions * 4 features (wall/body/green/red inverse distance)
            self.input_size = 16

        # Network architecture defaults:
        #   single — [448, 320, 64] from Optuna tuning (study snake-single-10).
        #   multi  — [64, 64], unchanged (matches the 16-D feature input).
        # Optuna tuning can still override both shape and dropout.
        if hidden_layers is None:
            hidden_layers = (
                [448, 320, 64] if mode == 'single' else [64, 64]
            )
        if dropout_rate is None:
            dropout_rate = 0.1 if mode == 'single' else 0.0
        self.hidden_layers = list(hidden_layers)
        self.dropout_rate = dropout_rate

        self.optimizer = optimizers.Adam(
            learning_rate=learning_rate, clipnorm=1.0
        )
        # Huber loss: quadratic for |error| <= 1.0, linear above.
        # Stable gradients when Q-value errors span a wide range at training
        # start, unlike MSE which produces exploding gradients (error^2) for
        # large targets.
        self.loss_fn = losses.Huber(delta=1.0)
        self.debug = debug
        self.train_count = 0

        # Experience replay buffer.
        # 150K default: at ~200 steps/episode covers ~750 episodes, retaining
        # useful experiences across the full exploitation phase.
        self.memory = deque(maxlen=memory_size)
        # Batch 512 default: lower gradient variance when Q-targets span a
        # wide range. TF handles larger batches efficiently.
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency

        # Main network (trained every step) and target network (synced
        # periodically)
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        """
        Build and return the neural network from the configured
        ``hidden_layers`` and ``dropout_rate``.

        Architecture is a stack of Dense(ReLU) layers, optionally with
        Dropout in between (no dropout before the output). The last layer
        is Dense(output_size, linear) producing one Q-value per action.
        """
        layer_list = [layers.Input(shape=(self.input_size,))]
        for i, units in enumerate(self.hidden_layers):
            layer_list.append(layers.Dense(units, activation='relu'))
            # Dropout between hidden layers, never before the output
            if (
                self.dropout_rate > 0
                and i < len(self.hidden_layers) - 1
            ):
                layer_list.append(layers.Dropout(self.dropout_rate))
        layer_list.append(
            layers.Dense(self.output_size, activation='linear')
        )
        return Sequential(layer_list)

    def update_target_model(self):
        """Copy weights from the main model to the target model."""
        self.target_model.set_weights(self.model.get_weights())

    def set_learning_rate(self, lr: float):
        """
        Update the optimizer learning rate without retracing the graph.

        Uses tf.Variable.assign() so the compiled _train_step graph is not
        invalidated — zero retrace cost. Called each episode to coordinate
        learning rate spikes with exploration cycles.
        """
        self.optimizer.learning_rate.assign(lr)

    def process_view(self, view, board_size=None):
        """
        Convert the raw 4-direction view into a network-ready feature tensor.

        Dispatches based on the agent's mode:
          - 'single': flattened one-hot encoding of shape (1, 4*board_size*6)
          - 'multi' : size-invariant 16-D feature vector

        The board_size argument is only used in 'single' mode; pass None for
        'multi'. If omitted in 'single' mode, falls back to self.board_size.
        """
        if self.mode == 'multi':
            return self._process_view_features(view)
        return self._process_view_onehot(view, board_size or self.board_size)

    def _process_view_onehot(self, view, board_size):
        """
        Single-mode encoder: flattened one-hot of the full visible cross.

        Each cell in each arm becomes a 6-element vector with a 1 at the
        category index. If an arm reaches board_size cells without hitting a
        wall, a virtual wall is placed at the last cell so the agent always
        perceives a boundary.
        """
        total_cells = 4 * board_size
        one_hot = np.zeros(
            (total_cells, self.num_categories), dtype=np.float32
        )

        cell_idx = 0
        for arm in view:
            arm_cells = list(arm[:board_size])

            # Virtual wall: if no wall is detected within the visible range,
            # mark the last cell as a wall to simulate the board boundary
            if len(arm_cells) == board_size and 'W' not in arm_cells:
                arm_cells[-1] = 'W'

            for c in arm_cells:
                if cell_idx < total_cells:
                    cat_idx = CHAR_MAP.get(str(c).strip()[-1], 0)
                    one_hot[cell_idx, cat_idx] = 1.0
                    cell_idx += 1

            # Pad remaining cells in this arm if it is shorter than board_size
            remaining = board_size - (cell_idx % board_size)
            if remaining < board_size:
                cell_idx += remaining

        return one_hot.flatten().reshape(1, -1)

    def _process_view_features(self, view):
        """
        Multi-mode encoder: size-invariant 16-D feature vector.

        For each of the 4 directions (UP, DOWN, LEFT, RIGHT) emits 4 numbers:
          [d_wall, d_body, d_green, d_red]
        where each is 1/distance to the nearest occurrence of that entity in
        that arm (1.0 = adjacent, smaller = farther, 0.0 = not visible).

        This representation has no dependence on the board size, so a network
        trained on it can play any board it was exposed to during training.
        """
        features = np.zeros(16, dtype=np.float32)

        for arm_idx, arm in enumerate(view):
            wall_d = body_d = green_d = red_d = 0  # 0 means "not seen"
            for cell_idx, cell in enumerate(arm):
                char = str(cell).strip()[-1] if cell else '0'
                distance = cell_idx + 1  # 1-based: closest cell is distance 1
                if char == 'W' and wall_d == 0:
                    wall_d = distance
                elif char == 'S' and body_d == 0:
                    body_d = distance
                elif char == 'G' and green_d == 0:
                    green_d = distance
                elif char == 'R' and red_d == 0:
                    red_d = distance

            base = arm_idx * 4
            features[base + 0] = 1.0 / wall_d if wall_d > 0 else 0.0
            features[base + 1] = 1.0 / body_d if body_d > 0 else 0.0
            features[base + 2] = 1.0 / green_d if green_d > 0 else 0.0
            features[base + 3] = 1.0 / red_d if red_d > 0 else 0.0

        return features.reshape(1, -1)

    def get_action(self, view, epsilon=0):
        """
        Choose an action using an epsilon-greedy policy.

        With probability epsilon, return a random action; otherwise return the
        action with the highest predicted Q-value.
        """
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.output_size)

        state = self.process_view(view, self.board_size)
        q_values = self.model(state, training=False)
        return int(np.argmax(q_values[0]))

    def remember(self, state_view, action, reward, next_state_view, done):
        """Store a transition in the replay memory."""
        self.memory.append((state_view, action, reward, next_state_view, done))

    @tf.function
    def _train_step(self, states, actions, targets):
        """Perform one gradient update step on the main network."""
        with tf.GradientTape() as tape:
            all_q = self.model(states, training=True)
            masks = tf.one_hot(actions, self.output_size)
            current_q = tf.reduce_sum(all_q * masks, axis=1)
            loss = self.loss_fn(targets, current_q)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables)
        )
        return loss

    def replay(self):
        """
        Sample a mini-batch from replay memory and train the main network.

        Uses the target network to compute stable Q-value targets.
        Periodically syncs the target network with the main network.

        Returns the training loss, or None if the memory is too small.
        """
        if len(self.memory) < self.batch_size:
            return None

        minibatch = random.sample(self.memory, self.batch_size)

        states, next_states, actions, rewards, dones = [], [], [], [], []
        for sv, a, r, nsv, d in minibatch:
            states.append(self.process_view(sv, self.board_size)[0])
            next_states.append(self.process_view(nsv, self.board_size)[0])
            actions.append(a)
            rewards.append(r)
            dones.append(d)

        states = np.array(states)
        next_states = np.array(next_states)
        actions = np.array(actions, dtype=np.int32)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        # Double DQN Bellman target:
        #   r + gamma * Q_target(s', argmax_a Q_main(s', a))
        # Standard DQN uses the same (target) network to both select and
        # evaluate the best next action, which systematically overestimates
        # Q-values and causes the agent to be falsely confident, producing
        # noisy conflicting Q-targets. Double DQN separates the two roles:
        # the main network selects the action, the target network evaluates
        # it — eliminating the overestimation bias.
        next_q_main = self.model(next_states, training=False).numpy()
        best_actions = np.argmax(next_q_main, axis=1)
        next_q_target = self.target_model(next_states, training=False).numpy()
        max_next_q = next_q_target[
            np.arange(len(best_actions)), best_actions
        ]
        targets = rewards + (1 - dones) * self.gamma * max_next_q

        loss = self._train_step(states, actions, targets)
        self.train_count += 1

        # Sync target network at fixed intervals
        if self.train_count % self.target_update_frequency == 0:
            self.update_target_model()
            if self.debug:
                print(
                    f"Target model updated at training step "
                    f"{self.train_count}"
                )

        return loss.numpy()
