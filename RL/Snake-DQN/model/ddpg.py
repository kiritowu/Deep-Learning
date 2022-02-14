from typing import Tuple, Optional, List
from git import typ

import gym
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import wandb
from tensorflow.keras import Sequential
from tensorflow.keras import initializers
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from utils import ReplayBuffer, Experience, OUActionNoise


class DDPG:
    def __init__(
        self,
        env: gym.Env,
        tau: float = 0.001,
        gamma: float = 0.99,
        lr_act: float = 1e-4,
        lr_cric: float = 1e-3,
        auc_std: float = 0.2,
        log_wandb: bool = False,
        is_continuous: bool = True,
        replay_buffer: Optional[ReplayBuffer] = None,
        actor_layers: Optional[List[int]] = None,
        critic_layers: Optional[List[int]] = None,
    ) -> None:
        """
        Construct a 'Deep Deterministic Policy Gradient' object that uses
        actor-critics framework to choosing the optimal policy.

        Reference:
        https://keras.io/examples/rl/ddpg_pendulum/
        https://github.com/CUN-bjy/gym-ddpg-keras

        """
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.log_wandb = log_wandb

        self.lr_act = lr_act
        self.lr_cric = lr_cric
        self.tau = tau
        self.gamma = gamma
        self.rewards_list = []
        self.is_continuous = is_continuous

        self.buffer = replay_buffer if replay_buffer else ReplayBuffer(maxlen=100_000)
        self.batch_size = 64

        # tuple that describe the discrete / continuous action space available
        # => 2 if continuous else 3 for Lunar-Lander
        self.num_action_space = (
            self.env.action_space.shape[0] if is_continuous else self.env.action_space.n
        )
        # a set of values reflective of the environment state that the agent has
        # access to
        # => 8 for Lunar-Lander-V2
        self.num_observation_space = env.observation_space.shape[0]

        # Add OrnsteinUhlenbeckProcess Noise to solve Exploration-Exploitation Trade-offs
        self.noise = OUActionNoise(
            mean=np.zeros(self.num_action_space),
            std_deviation=float(auc_std) * np.ones(self.num_action_space),
        )

        self.actor_layers = [256, 64] if not actor_layers else actor_layers
        self.critic_layers = [256, 64] if not critic_layers else critic_layers

        self.actor = self.initialize_actor_model("actor")
        self.critic = self.initialize_critic_model("critic")
        self.actor_target = tf.keras.models.clone_model(self.actor)
        self.critic_target = tf.keras.models.clone_model(self.critic)

        self.actor_optimizer = Adam(self.lr_act)
        self.critic_optimizer = Adam(self.lr_cric)

    def initialize_actor_model(self, name: str) -> tf.keras.Model:
        """
        Build the actor network which takes in the state and returns action as output.

        :param name: Name to the actor network
        """
        actor_net = Sequential(name=name)
        # Make a copy of the layers list
        layers = self.actor_layers[:]
        # Build Actor Network
        first_layer_actor = layers.pop(0)
        # Input Layer
        actor_net.add(
            Dense(
                first_layer_actor,
                input_dim=self.num_observation_space,
                activation="relu",
                kernel_initializer=initializers.random_uniform(
                    -1 / np.sqrt(self.num_action_space),
                    1 / np.sqrt(self.num_action_space),
                ),
            )
        )
        # Hidden layers
        for layer in layers:
            actor_net.add(
                Dense(
                    layer,
                    activation="relu",
                    kernel_initializer=initializers.random_uniform(
                        -1 / np.sqrt(self.num_action_space),
                        1 / np.sqrt(self.num_action_space),
                    ),
                )
            )
        # Output Layer
        # Tanh is used to limit action-space to [-1, 1]
        actor_net.add(
            Dense(
                self.num_action_space,
                activation="tanh",
                kernel_initializer=initializers.random_uniform(
                    -3e-3, 3e-3  # Close to 0 for earlier episodes
                ),
            )
        )

        return actor_net

    def initialize_critic_model(self, name: str) -> tf.keras.Model:
        """
        Build the critic network which takes in the state and action and return
         the estimated q-value.

        :param name: Name to the critic network
        """
        critic_net = Sequential(name=name)
        # Make a copy of the layers list
        layers = self.critic_layers[:]
        # Build Critics Network
        first_layer_critic = layers.pop(0)
        # Input Layer
        critic_net.add(
            Dense(
                first_layer_critic,
                # InputDim = 8(state)+2(action)
                input_dim=self.num_observation_space + self.num_action_space,
                activation="relu",
                kernel_initializer=initializers.random_uniform(
                    -1 / np.sqrt(self.num_action_space),
                    1 / np.sqrt(self.num_action_space),
                ),
            )
        )
        # Hidden layers
        for layer in layers:
            critic_net.add(
                Dense(
                    layer,
                    activation="relu",
                    kernel_initializer=initializers.random_uniform(
                        -1 / np.sqrt(self.num_action_space),
                        1 / np.sqrt(self.num_action_space),
                    ),
                )
            )
        # Output Layer
        # Return the estimated Q-value given the state and action
        critic_net.add(Dense(1, activation="linear"))

        return critic_net

    def get_action(self, state: np.ndarray) -> np.ndarray:
        """
        Use actor network to generate the action along with added noise
        for the sake of exploration.
        """
        action = self.actor.predict(state)[0]

        action_with_noise = action + self.noise()

        # Bounded by action space in [-1,1] if continuous else pick action with maximum q
        return (
            np.clip(action_with_noise, -1, 1)
            if self.is_continuous
            else np.argmax(action_with_noise)
        )

    def train(self, num_episodes=1000, mean_stopping=True):
        # iterate over the number of episodes
        for episode in range(num_episodes):

            state = self.env.reset()
            reward_for_episode = 0
            max_num_steps = 1000
            state = state.reshape(1, self.num_observation_space)

            for step in range(max_num_steps):
                # get the action for the current state
                action = self.get_action(state)

                # get the next_state, reward, done and info after running the action
                next_state, reward, done, info = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.num_observation_space])

                # store the experience in replay memory
                self.buffer.append(Experience(state, action, reward, next_state, done))

                # add up rewards
                reward_for_episode += reward
                state = next_state

                # train dqn
                self.update_weights()

                if done:
                    break

            self.rewards_list.append(reward_for_episode)

            # check for terminal condition
            last_rewards_mean = np.mean(self.rewards_list[-100:])
            if last_rewards_mean > 280 and mean_stopping:
                print("DQN Training Complete...")
                break

            print(
                "[{:0>3}] Reward: {: >8.3f} | Avg Reward: {: >8.3f}".format(
                    episode, reward_for_episode, last_rewards_mean
                )
            )
            if self.log_wandb:
                wandb.log(
                    {
                        "Episode": episode,
                        "Reward": reward_for_episode,
                        "Avg-Reward-100e": last_rewards_mean,
                    }
                )

    def update_weights(self):
        # buffer size check
        if len(self.buffer) < self.batch_size:
            return

        # prevent over-training: stop learning if the mean of the latest 10 rewards is greater than 180
        # change 180 -> 200
        # this causes "RuntimeWarning: Mean of empty slice."
        if np.mean(self.rewards_list[-10:]) > 200:
            return

        # randomly sample a replay memory with the size of the batch
        # getting the states, actions, rewards, next_state and done_list from the random sample
        states, actions, rewards, next_states, done_list = self.buffer.sample(
            self.batch_size
        )

        # Train critic network
        self._train_critic(states, actions, rewards, next_states, done_list)
        # Train actor network
        self._train_actor(states)

        # Update Target Networks with tau (Exponential Weight Average)
        # target_weight = tau*actual_weight + (1-tau)*target_weight
        self._update_target(self.actor_target.variables, self.actor.variables)
        self._update_target(self.critic_target.variables, self.critic.variables)

    @tf.function
    def _train_critic(self, states, actions, rewards, next_states, done_list):
        """
        Train Critic Network with Loss Function of TD-Error.
        # Loss Function : Minimize the TD-Error Generated with Following Formula
        # TD_Error = Expected Return - Estimated Return
        #          = (reward + gamma * next_q_from_target_critic) - q_from_critic
        """
        with tf.GradientTape() as tape:
            # Calculate the Expected Loss using Target Critic and Actor
            # Generate next actions and qs using target networks
            next_target_actions = self.actor_target(next_states, training=True)
            next_target_qs = self.critic_target(
                K.concatenate([next_states, next_target_actions], axis=1),
                training=True
            )
            target_q = rewards + self.gamma * next_target_qs * (1. - done_list)
            q_values = self.critic(
                K.concatenate([states, actions], axis=1), training=True
            )
            td_error = q_values - target_q
            critic_loss = tf.reduce_mean(tf.math.square(td_error))

        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic.trainable_variables)
        )

    @tf.function
    def _train_actor(self, states):
        """
        Train Actor Network with Loss Function to Maximise the Q-Value.
        # Loss Function: Maximise the mean Q-value generated by the critic network
        """
        with tf.GradientTape() as tape:
            actions = self.actor(states, training=True)
            q_values = self.critic(
                K.concatenate([states, actions], axis=1), training=True
            )
            actor_loss = -tf.reduce_mean(q_values)

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor.trainable_variables)
        )

    @tf.function
    def _update_target(self, target_weights, weights):
        for target_weight, weight in zip(target_weights, weights):
            target_weight.assign(weight * self.tau + (1 - self.tau) * target_weight)

    def save(self, path_to_checkpoint: str):
        self.actor.save(f"{path_to_checkpoint}/actor")
        self.critic.save(f"{path_to_checkpoint}/critic")

    def load(self, path_to_checkpoint: str):
        self.actor.load_weights(f"{path_to_checkpoint}/actor")
        self.actor_target.load_weights(f"{path_to_checkpoint}/actor")
        self.critic.load_weights(f"{path_to_checkpoint}/critic")
        self.critic_target.load_weights(f"{path_to_checkpoint}/critic")

    def train_actor_only(
        self, critic_network: tf.keras.Model, num_episodes, mean_stopping=True
    ):
        assert not self.is_continuous, "The environment needs to be continuous"
        self.critic = tf.keras.models.clone_model(critic_network)
        self.critic_target = tf.keras.models.clone_model(self.critic)

        for episode in range(num_episodes):

            state = self.env.reset()
            reward_for_episode = 0
            max_num_steps = 1000
            state = state.reshape(1, self.num_observation_space)

            for step in range(max_num_steps):
                # get the action for the current state
                action = self.get_action(state)

                # get the next_state, reward, done and info after running the action
                next_state, reward, done, info = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.num_observation_space])

                # store the experience in replay memory
                self.buffer.append(Experience(state, action, reward, next_state, done))

                # add up rewards
                reward_for_episode += reward
                state = next_state

                # train dqn
                self.update_actors_weights_only()

                if done:
                    break

            self.rewards_list.append(reward_for_episode)

            # check for terminal condition
            last_rewards_mean = np.mean(self.rewards_list[-100:])
            if last_rewards_mean > 280 and mean_stopping:
                print("DQN Training Complete...")
                break

            print(
                "[{:0>3}] Reward: {: >8.3f} | Avg Reward: {: >8.3f}".format(
                    episode, reward_for_episode, last_rewards_mean
                )
            )

    def update_actors_weights_only(self):
        # buffer size check
        if len(self.buffer) < self.batch_size:
            return

        # prevent over-training: stop learning if the mean of the latest 10 rewards is greater than 180
        # change 180 -> 200
        # this causes "RuntimeWarning: Mean of empty slice."
        if np.mean(self.rewards_list[-10:]) > 200:
            return

        # randomly sample a replay memory with the size of the batch
        # getting the states, actions, rewards, next_state and done_list from the random sample
        states, actions, rewards, next_states, done_list = self.buffer.sample(
            self.batch_size
        )

        # Train actor network
        with tf.GradientTape() as tape:
            # Clever argmax trick to make the operation differentiable
            discrete_actions_matrix = argmax_diff(self.actor(states, training=True))
            # Use Ele-wise multiplication for differentiable slicing
            q_values = tf.math.multiply(self.critic(states), discrete_actions_matrix)
            # Gradient Ascend to get highest q_values based on critic network
            actor_loss = -tf.reduce_mean(q_values)

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor.trainable_variables)
        )


def argmax_diff(x, epsilon=1e-6):
    """
    Differentiable Argmax Inspired By
    https://www.titanwolf.org/Network/q/460c8510-e49c-4099-abd6-06013d4a1b45/y
    """
    max_value = tf.reduce_max(x, axis=1, keepdims=True)  # [[5],[7]]
    clip_min = max_value - epsilon  # [[4.99],[6.99]]
    one_hot_pos = (tf.clip_by_value(x, clip_min, max_value) - clip_min) / (
        max_value - clip_min
    )
    return one_hot_pos