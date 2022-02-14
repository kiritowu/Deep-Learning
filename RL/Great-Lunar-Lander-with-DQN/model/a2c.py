from typing import List, Optional, Tuple

import gym
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import wandb
from tensorflow.keras import Model
from tensorflow.keras import layers as tfl
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm


class A2C:
    def __init__(
        self,
        env: gym.Env,
        lr: float,
        gamma: float = 0.99,
        entropy_beta: float = 0.01,
        reward_steps: int = 4,
        clip_grad: float = 0.1,
    ):
        """Build an Advantage Actor Critic (A2C) model, a synchronous deterministic version of A3C. It is an actor critic method, which uses a Critic model to estimate the optimal state-value function V(s), so as to calculate a normalized action-value function, called an Advantage Function. This advantage function is used to calculate policy gradients which are used to update an Actor network learning to estimate the optimal policy.

        Implementation inspired by:
        - https://keras.io/examples/rl/actor_critic_cartpole/
        - https://github.com/germain-hug/Deep-RL-Keras
        - https://arxiv.org/pdf/1602.01783.pdf
        - Lapan, 2020, Deep Reinforcement Learning Hands On

        :param env: LunarLander Environment
        :type env: gym.Env
        :param lr: Learning rate of optimizers
        :type lr: float
        :param gamma: Discount factor for expected return
        :type gamma: float
        :param entropy_beta: Determines importance of Entropy loss, defaults to 0.01
        :type entropy_beta: float, optional
        :param reward_steps: Number of steps N to play out when calculating rewards, defaults to 4
        :type reward_steps: int, optional
        :param clip_grad: Gradient clipping is done to prevent gradients from becoming too large, causing a large shift in policy, defaults to 0.1
        :type clip_grad: float, optional
        """
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

        self.lr = lr
        self.gamma = gamma
        self.entropy_beta = entropy_beta
        self.reward_steps = reward_steps
        self.clip_grad = clip_grad

        self.state_shape = self.observation_space.shape[0]
        self.num_actions = self.action_space.n

        self.a2c_model = self.initialize_actor_critic("A2C")
        self.optimizer = Adam(learning_rate=lr)
        self.huber_loss = tf.keras.losses.Huber()

        self.reward_history = []  # Keep track of rewards per episode

    def initialize_actor_critic(self, name: str) -> tf.keras.Model:
        """Create Actor Critic Networks. Both actor and critic networks share the same parameters for the initial layers as the low level features should be common between the both of them."

        :param name: Model name
        :type name: str
        :return: Actor Critic Model
        :rtype: tf.keras.Model
        """
        # Input
        inputs = tfl.Input(
            shape=(self.state_shape)
        )  # Both networks take in the state of the environment as the input

        # Followed by 2 FC Layers
        x = tfl.Dense(16, activation="relu")(inputs)
        x = tfl.Dense(16, activation="relu")(x)
        # Then, define separate layers for the actor and critic

        actor = tfl.Dense(32, activation="relu")(x)
        actor = tfl.Dense(64, activation="relu")(actor)
        actor = tfl.Dense(self.num_actions, activation="softmax")(actor)
        # Estimate policy
        critic = tfl.Dense(32, activation="relu")(x)
        critic = tfl.Dense(64, activation="relu")(critic)
        critic = tfl.Dense(1)(critic)  # Estimate the state value function)

        model = Model(inputs=inputs, outputs=[actor, critic], name=name)
        print(model.summary())
        return model

    def train(self, num_episodes: int = 1000, mean_stopping: bool = True):
        """Trains A2C (Advantage Actor Critic) Model, which follows this algorithm for each training episode
        1. Play N steps in the environment using the current policy, and saving the state, action and reward
        2. Initialize a variable to hold the reward, setting it as 0 if the state is terminal, otherwise setting it as the current value of the state.
        3. Iterating through the time steps in reverse,
            - Set the reward for each time step as a discounted reward (current reward + discounted future reward)
            - Accumulate the policy gradients (using the calculated Advantage: Advantage = Value - Mean Value)
            - Accumulate the value gradients
        4. Update the parameters of the networks based on the gradients, moving in the direction of the policy gradients and in the opposite direction of the value gradients
        5. Stop when the environment is "solved" (i.e. a score of 200 in the case of the Lunar Lander)

        Notes: To encourage exploration of the environment, I add an entropy loss which can be minimized when the probability distribution of the policy is uniform.

        :param num_episodes: [description], defaults to 1000
        :type num_episodes: int, optional
        :param mean_stopping: [description], defaults to True
        :type mean_stopping: bool, optional
        :raises NotImplementedError: [description]
        """
        tqdm_e = tqdm(
            range(num_episodes), total=num_episodes, desc="Score", unit=" episodes"
        )
        for episode in tqdm_e:
            # Reset Actions, State, Rewards
            state = self.env.reset()
            state_values = []  # Keep track of Critic predictions for optimization
            actions, rewards = [], []
            cumulative_reward = 0

            # Initialize
            with tf.GradientTape() as tape:
                for t in range(1, self.reward_steps):
                    # Play n steps in the environment
                    action_probs, critic_value = self.forward(state)

                    # To select an action, sample from the probability distribution
                    action = np.random.choice(
                        self.num_actions, p=np.squeeze(action_probs)
                    )

                    actions.append(tf.math.log(action_probs[0, action]))
                    state_values.append(critic_value[0, 0])

                    # Step through that action
                    state, reward, done, _ = self.env.step(action)
                    cumulative_reward += reward
                    rewards.append(reward)
                    if done:
                        break

                # Check for terminal conditions
                self.reward_history.append(cumulative_reward)
                last_rewards_mean = np.mean(
                    self.reward_history[-100:]
                )  # Average Reward over 100 Episode Window
                if last_rewards_mean > 280 and mean_stopping:
                    print("Training Complete")
                    break

                # Calculate expected return
                returns = []
                expected_return = 0
                for r in reversed(rewards):
                    expected_return = r + self.gamma * expected_return
                    returns.insert(0, expected_return)
                returns = np.array(returns)
                # Calculate Advantage A(s, a) = Q(s, a) - V(s)
                eps = 1e-8 # Small numerical constant to prevent division by zero
                advantages = returns - np.mean(returns) / (np.std(returns) + eps)
                advantages = advantages.tolist()

                # Calculate Loss for Gradient Updates
                history = zip(actions, state_values, advantages)
                actor_losses = []
                entropy_losses = []
                critic_losses = []

                for (
                    log_prob,
                    value,
                    advantage,
                ) in history:
                    # Actor Loss
                    actor_loss = (
                        advantage - value
                    )  # What was the difference between the predicted advantage and the actual advantage?
                    actor_loss *= log_prob  # We want to update the actor so that actions with higher advantage are predicted more often
                    actor_loss *= (
                        -1
                    )  # For the policy network, we need to perform Gradient Ascent, so we flip the sign so that the optimizer can optimize the actor loss

                    # Entropy Loss : probability * log probability
                    entropy_loss = -(
                        tf.math.exp(log_prob) * log_prob
                    )  # Entropy is used to encourage exploration of the environment by encouraging a uniform distribution of the policy

                    # Critic Loss
                    critic_loss = self.huber_loss(
                        tf.expand_dims(value, 0), tf.expand_dims(advantage, 0)
                    )  # Huber loss is used instead of MSE, as it combines the benefits of MSE and MAE losses. Outliers are penalized (like in L2 Loss), and the network is encouraged to reduce the loss to 0 when loss is very small (like in L1 Loss)

                    actor_losses.append(actor_loss)
                    entropy_losses.append(entropy_loss)
                    critic_losses.append(critic_loss)

                # Overall Loss
                loss = (
                    sum(actor_losses)
                    + sum(critic_losses)
                    + self.entropy_beta * sum(entropy_losses)
                )

                # Perform Backprop to Get Gradients
                gradients = tape.gradient(loss, self.a2c_model.trainable_variables)
                # Do Gradient Clipping
                if self.clip_grad is not None:
                    gradients = [
                        tf.clip_by_norm(gradient, self.clip_grad)
                        for gradient in gradients
                    ]

                self.optimizer.apply_gradients(
                    zip(gradients, self.a2c_model.trainable_variables)
                )
            wandb.log(
                {
                    "Episode": episode,
                    "Reward": cumulative_reward,
                    "Avg-Reward-100e": last_rewards_mean,
                    "Loss": loss,
                    "Actor Loss": sum(actor_losses),
                    "Critic Loss": sum(critic_losses),
                    "Entropy Loss": sum(entropy_losses),
                }
            )
            tqdm_e.set_description(f"Score: {cumulative_reward}")
            tqdm_e.refresh()
        # Save Model
        self.save("saved-models/a2c/")
        wandb.log_artifact("saved-models/a2c/", name="A2C", type="model")

    @tf.function
    def forward(self, x: np.ndarray):
        """Return policy and value given state.

        :param x: Current state of the environment
        :type x: np.ndarray
        :return: Softmax Probabilities of Actions and Estimated Value Given State
        :rtype: Tuple
        """
        x = tf.convert_to_tensor(x)
        x = tf.expand_dims(x, 0)
        action_probs, critic_value = self.a2c_model(x)
        return action_probs, critic_value

    def save(self, path: str):
        self.a2c_model.save(path)

    def load(self, path: str):
        self.a2c_model.load_weights(path)
