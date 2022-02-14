from typing import List, Optional

import numpy as np
import gym
import wandb

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import clone_model

from utils import ReplayBuffer, Experience, NoisyDense

class DQN:
    def __init__(
        self,
        env:gym.Env,
        lr:float,
        gamma:float,
        epsilon:float,
        epsilon_decay:float,
        target_update_interval: int = 100,
        log_wandb: bool=False,
        tuning_condition: bool=False,
        replay_buffer:Optional[ReplayBuffer]=None,
        layers:Optional[List[int]]=None,
        save_path: str = "./saved-models/dqn.h5"
    ):
        """
        Construct a new 'Deep Q-Network' object.

        :param env: The environment of the game
        :param lr: The learning rate of the agent
        :param gamma: The amount of weight it gives to future rewards in the value function
        :param epsilon: The probability where we do not go with the “greedy” action with the highest Q-value but rather choose a random action
        :param epsilon_decay: The rate by which epsilon decreases after an episode
        :param target_update_interval: The interval between updates of the target network
        :param replay_buffer: Replay memory object to store and sample observations from for training.
        :param tuning_condition: If true, the training will stop when the condition is met.
         Defaults to double-end queue with maximum length of 500_000 steps.
        """
        self.log_wandb = log_wandb
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.save_path = save_path

        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.target_update_interval = target_update_interval

        self.rewards_list = []

        # store trajectories of experience when executing a policy in an environment
        self.buffer = replay_buffer if replay_buffer else ReplayBuffer(maxlen=500_000)
        self.batch_size = 64
        self.epsilon_min = 0.01
        # agents have either a dis- crete or a continuous action space
        self.num_action_space = self.action_space.n
        # a set of values reflective of the environment state that the agent has access to
        self.num_observation_space = env.observation_space.shape[0]

        self.layers = [512, 256] if not layers else layers
        assert len(self.layers) >= 1, "You need at least one hidden layer"

        self.model = self.initialize_model()
        self.model_target = clone_model(self.model)

        # Track the hyperparameters
        wandb.config.update({
            "lr": self.lr,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "target_update_interval": self.target_update_interval,
            "batch_size": self.batch_size,
            "layers": self.layers,
        })

        self.tuning_condition = tuning_condition

    def initialize_model(self):
        model = Sequential()

        # the number of starting neurons is equal to the number of observation space
        layer = self.layers[:] # Make a copy
        first_layer = layer.pop(0)
        model.add(Dense(first_layer, input_dim=self.num_observation_space, activation="relu"))
        
        # Hidden Layers
        for layer in layer:
            model.add(Dense(layer, activation="relu"))

        # the number of ending neurons is equal to the number of action space
        model.add(Dense(self.num_action_space, activation="linear"))

        # Compile the model with MSE of TD-Error with Adam
        model.compile(loss="mean_squared_error", optimizer=Adam(learning_rate=self.lr))
        return model

    def get_action(self, state):
        # a random action is chosen when a random chosen number is lower than the epsilon
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()

        # if not, the model will predict the action with its current state
        predicted_actions = self.model.predict(state)

        # returns the index of the actions with the highest score
        return np.argmax(predicted_actions[0])

    def update_weights(self):

        # buffer size check
        if len(self.buffer) < self.batch_size:
            return

        # # prevent over-training: stop learning if the mean of the latest 10 rewards is greater than 180
        # # this causes "RuntimeWarning: Mean of empty slice."
        # if np.mean(self.rewards_list[-10:]) > 200:
        #     return

        # randomly sample a replay memory with the size of the batch
        # getting the states, actions, rewards, next_state and done_list from the random sample
        states, actions, rewards, next_states, done_list = self.buffer.sample(self.batch_size, dqn=True)

        # calculate the loss to create a target vector for the model to fit with the states
        targets = rewards + self.gamma * (np.amax(self.model_target.predict_on_batch(next_states), axis=1)) * (1 - done_list)
        target_vec = self.model.predict_on_batch(states)
        indexes = np.array([i for i in range(self.batch_size)])
        target_vec[[indexes], [actions]] = targets

        # fit the model with the states and the target vector for one iteration
        self.model.fit(states, target_vec, epochs=1, verbose=0)

    def _update_target(self, target_weights, weights):
        for target_weight, weight in zip(target_weights, weights):
            target_weight.assign(weight)

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
                if isinstance(action, tf.Tensor):
                    action = action.numpy()
                
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

                # Every k steps, copy actual network weights to the target network weights
                if (step + 1) % self.target_update_interval == 0:
                    self._update_target(self.model_target.variables, self.model.variables)

                if done: break
            
            self.rewards_list.append(reward_for_episode)

            # decay the epsilon after each episode
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            # check for terminal condition
            last_rewards_mean = np.mean(self.rewards_list[-100:])
            if last_rewards_mean > 280 and mean_stopping:
                print("DQN Training Complete...")
                break

            if self.tuning_condition and (episode > 50 and last_rewards_mean < -200):
                print(f"Training stopped early: Episode {episode} with {last_rewards_mean:.3f} mean reward")
                break
            
            print("[{:0>3}] Reward: {: >8.3f} | Avg Reward: {: >8.3f} | e: {:.3f}"
                  .format(episode, reward_for_episode, last_rewards_mean, self.epsilon))
            
            if self.log_wandb:
                wandb.log({
                    "Episode": episode,
                    "Reward": reward_for_episode,
                    "Avg-Reward-100e": last_rewards_mean,
                    "Epsilon": self.epsilon,
                    "Episode Length": step
                })
        
        if self.log_wandb:
            wandb.finish()
        
        if not self.tuning_condition:
            self.save(self.save_path)

    def save(self, path:str):
        self.model.save(path)