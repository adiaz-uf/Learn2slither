"""
Agent module.

Implements the Deep Q-Network (DQN) agent with experience replay and a
target network. The agent takes the snake's cross-shaped vision as input
and outputs Q-values for the four possible actions
(UP=0, DOWN=1, LEFT=2, RIGHT=3).

The state encoder is a 16-D size-invariant feature vector: for each of the
4 cardinal directions, four numbers
  [d_wall, d_body, d_green, d_red]
where each is 1/distance to the nearest occurrence of that entity along the
arm (1.0 = adjacent, smaller = farther, 0.0 = not visible). The same network
plays any board size (10/14/18) without architectural changes.
"""

import numpy as np
import tensorflow as tf
from keras import Sequential, layers, optimizers, losses
from collections import deque
import random


class Agent:
    """
    DQN agent with experience replay and a periodically updated target
    network. Input is a 16-D feature vector regardless of board size; the
    output is a 4-element vector of Q-values, one per action.
    """

    INPUT_SIZE = 16

    def __init__(self, board_size=None, mode='single',
                 learning_rate=0.000216, gamma=0.9646, debug=False,
                 batch_size=512, target_update_frequency=1000,
                 memory_size=150000,
                 hidden_layers=None, dropout_rate=None):
        if mode not in ('single', 'multi'):
            raise ValueError(
                f"Unknown mode '{mode}'. Use 'single' or 'multi'."
            )

        # `mode` and `board_size` are informational only — they no longer
        # affect the network architecture. They label the training regime
        # (`single` = fixed board, `multi` = sampled boards) and tell the
        # UI which size selector to show.
        self.mode = mode
        self.board_size = board_size
        self.gamma = gamma
        self.output_size = 4  # UP, DOWN, LEFT, RIGHT

        self.input_size = self.INPUT_SIZE

        # Network architecture defaults: a small MLP suited to the 16-D
        # dense feature input. Optuna tuning can override both the shape
        # and the dropout rate.
        if hidden_layers is None:
            hidden_layers = [64, 64]
        if dropout_rate is None:
            dropout_rate = 0.0
        self.hidden_layers = list(hidden_layers)
        self.dropout_rate = dropout_rate

        self.optimizer = optimizers.Adam(
            learning_rate=learning_rate, clipnorm=1.0
        )
        # Huber loss: quadratic for |error| <= 1.0, linear above. Stable
        # gradients when Q-value errors span a wide range at training start,
        # unlike MSE which produces exploding gradients (error^2) for large
        # targets.
        self.loss_fn = losses.Huber(delta=1.0)
        self.debug = debug
        self.train_count = 0

        # Experience replay buffer: a circular FIFO of past transitions.
        # Sampling random mini-batches from it decorrelates updates and
        # lets each transition contribute to many gradient steps.
        self.memory = deque(maxlen=memory_size)
        # Mini-batch sample size for each training step.
        self.batch_size = batch_size
        # Steps between target-network syncs.
        self.target_update_frequency = target_update_frequency

        # Main network (trained every step) and target network (synced
        # periodically).
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
            # Dropout between hidden layers, never before the output.
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
        invalidated — zero retrace cost.
        """
        self.optimizer.learning_rate.assign(lr)

    def process_view(self, view, board_size=None):
        """
        Convert the raw 4-direction view into a 16-D network input tensor.

        ``board_size`` is accepted only for backwards-compatible call
        signatures; it is ignored because the encoder is size-invariant.
        """
        del board_size  # unused
        return self._process_view_features(view)

    def _process_view_features(self, view):
        """
        Size-invariant 16-D feature encoder.

        For each of the 4 directions (UP, DOWN, LEFT, RIGHT) emits 4 numbers:
          [d_wall, d_body, d_green, d_red]
        where each is 1/distance to the nearest occurrence of that entity
        in that arm (1.0 = adjacent, smaller = farther, 0.0 = not visible).
        """
        features = np.zeros(16, dtype=np.float32)

        for arm_idx, arm in enumerate(view):
            wall_d = body_d = green_d = red_d = 0  # 0 means "not seen"
            for cell_idx, cell in enumerate(arm):
                char = str(cell).strip()[-1] if cell else '0'
                # 1-based: the closest cell is distance 1.
                distance = cell_idx + 1
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

        With probability epsilon, return a random action; otherwise return
        the action with the highest predicted Q-value.
        """
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.output_size)

        state = self.process_view(view)
        q_values = self.model(state, training=False)
        return int(np.argmax(q_values[0]))

    def remember(self, state_view, action, reward, next_state_view, done):
        """Store a transition in the replay memory."""
        self.memory.append(
            (state_view, action, reward, next_state_view, done)
        )

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
            states.append(self.process_view(sv)[0])
            next_states.append(self.process_view(nsv)[0])
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
        # Q-values. Double DQN separates the two roles: the main network
        # selects the action, the target network evaluates it.
        # Student looks at s' and gives 4 Q-values per state.
        next_q_main = self.model(next_states, training=False).numpy()
        # Student picks the action it thinks is best in s', [batch_size]
        best_actions = np.argmax(next_q_main, axis=1)
        # Teacher (frozen copy) also looks at s' and gives its own 4
        # Q-values, [batch_size, 4]
        next_q_target = self.target_model(
            next_states, training=False
        ).numpy()
        # Teacher's score for the action the student picked
        # (Double DQN split), [batch_size]
        max_next_q = next_q_target[
            np.arange(len(best_actions)), best_actions
        ]
        # Correct answer to learn (Bellman):
        # reward now + discounted future (0 if done), [batch_size]
        targets = rewards + (1 - dones) * self.gamma * max_next_q

        # One gradient step: nudge student so Q(s, a) gets closer to
        # target, [batch_size]
        loss = self._train_step(states, actions, targets)
        self.train_count += 1

        # Sync target network at fixed intervals.
        if self.train_count % self.target_update_frequency == 0:
            self.update_target_model()
            if self.debug:
                print(
                    f"Target model updated at training step "
                    f"{self.train_count}"
                )

        return loss.numpy()
