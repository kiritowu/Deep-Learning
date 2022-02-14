import random
from typing import List, Optional, Tuple

import numpy as np
import gym
import wandb

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential, Input, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import clone_model

from utils import ReplayBuffer, Experience

class DQN:
    def __init__(
        self,
        env:gym.Env,
        combined_observation_space:Tuple[Tuple[int,int,int], int],
        lr:float,
        gamma:float,
        epsilon:float,
        epsilon_decay:float,
        target_update_interval: int = 100,
        log_wandb: bool=False,
        replay_buffer:Optional[ReplayBuffer]=None,
        fc_layers:Optional[List[int]]=None,
        conv_layers:Optional[List[int]]=None
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
         Defaults to double-end queue with maximum length of 500_000 steps.
        """
        self.log_wandb = log_wandb
        self.env = env
        self.action_space = env.action_space
        self.combined_observation_space = combined_observation_space

        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.target_update_interval = target_update_interval
        self.rewards_list = []

        # store trajectories of experience when executing a policy in an environment
        self.buffer = replay_buffer if replay_buffer else ReplayBuffer(maxlen=2_500)
        self.batch_size = 64
        self.epsilon_min = 0.01
        # agents have either a dis- crete or a continuous action space
        self.num_action_space = 4


        self.fc_layers = [128,128,128] if not fc_layers else fc_layers
        assert len(self.fc_layers) >= 1, "You need at least one hidden layer"

        self.conv_layers = [32, 64, 128] if not conv_layers else conv_layers
        assert len(self.conv_layers) >= 1, "You need at least one hidden layer"

        self.model = self.initialize_model()
        self.model_target = clone_model(self.model)

        # Track the hyperparameters
        if self.log_wandb:
            wandb.config.update({
                "lr": self.lr,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "epsilon_decay": self.epsilon_decay,
                "target_update_interval": self.target_update_interval,
                "batch_size": self.batch_size,
                "fc_layers": self.fc_layers
            })

    def initialize_model(self):
        conv_layers = self.conv_layers[:] # Make a copy
        first_conv_layer = conv_layers.pop(0)

        i1 = Input(shape=self.combined_observation_space[0])
        i2 = Input(shape=self.combined_observation_space[1])
        x = Conv2D(first_conv_layer,8,4, padding="same", activation="relu")(i1)
        for conv_layer in conv_layers:
            x = Conv2D(conv_layer,3,4,padding="same", activation="relu")(x)
        x = Flatten()(x)
        x = Concatenate(axis=1)([x,i2])
        
        layer = self.fc_layers[:] # Make a copy
        first_layer = layer.pop(0)
        
        x = Dense(first_layer, activation="relu")(x)
        
        # Hidden fc_layers
        for layer in layer:
            x = Dense(layer, activation="relu")(x)

        # the number of ending neurons is equal to the number of action space
        out = Dense(self.num_action_space, activation="linear")(x)
        
        model = Model(inputs = [i1, i2], outputs = out)
        # Compile the model with MSE of TD-Error with Adam
        model.compile(loss="mean_squared_error", optimizer=Adam(learning_rate=self.lr))
        return model

    def get_action(self, state):
        # a random action is chosen when a random chosen number is lower than the epsilon
        if np.random.rand() < self.epsilon:
            return random.randint(0,3)
        # if not, the model will predict the action with its current state
        predicted_actions = self.model.predict([tf.expand_dims(state[0], axis=0),tf.expand_dims(state[1], axis=0)])

        # returns the index of the actions with the highest score
        return np.argmax(predicted_actions[0])

    def update_weights(self):

        # buffer size check
        if len(self.buffer) < self.batch_size:
            return

        # randomly sample a replay memory with the size of the batch
        # getting the states, actions, rewards, next_state and done_list from the random sample
        states, actions, rewards, next_states, done_list = self.buffer.sample(self.batch_size, dqn=True)

        # calculate the loss to create a target vector for the model to fit with the states
        targets = rewards + self.gamma * (np.max(self.model_target.predict_on_batch([
            np.concatenate(next_states[0]).reshape(-1, *self.combined_observation_space[0]),
            np.concatenate(next_states[1]).reshape(-1, self.combined_observation_space[1])
        ]), axis=1)) * (1 - done_list)
        target_vec = self.model.predict_on_batch([
            np.concatenate(states[0]).reshape(-1, *self.combined_observation_space[0]),
            np.concatenate(states[1]).reshape(-1, self.combined_observation_space[1])
        ])
        indexes = np.array([i for i in range(self.batch_size)])
        target_vec[[indexes], [actions]] = targets

        # fit the model with the states and the target vector for one iteration
        self.model.fit([
            np.concatenate(states[0]).reshape(-1, *self.combined_observation_space[0]),
            np.concatenate(states[1]).reshape(-1, self.combined_observation_space[1])
        ], target_vec, epochs=1, verbose=0)

    def _update_target(self, target_weights, weights):
        for target_weight, weight in zip(target_weights, weights):
            target_weight.assign(weight)

    def train(self, num_episodes=1000, mean_stopping=True):

        # iterate over the number of episodes
        for episode in range(num_episodes):
            
            state = self.env.reset()
            reward_for_episode = 0
            max_num_steps = 1000
            
            for step in range(max_num_steps):
                # get the action for the current state
                action = self.get_action(state)
                if isinstance(action, tf.Tensor):
                    action = action.numpy()
                # get the next_state, reward, done and info after running the action
                next_state, reward, done, info = self.env.step(int(action))
                
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
            if last_rewards_mean > 250 and mean_stopping:
                print("DQN Training Complete...")
                break
            
            print("[{:0>3}] Reward: {: >8.3f} | Avg Reward: {: >8.3f} | e: {:.3f} | Episode Length: {:}"
                  .format(episode, reward_for_episode, last_rewards_mean, self.epsilon, step))
            
            if self.log_wandb:
                wandb.log({
                    "Episode": episode,
                    "Reward": reward_for_episode,
                    "Avg-Reward-100e": last_rewards_mean,
                    "Epsilon": self.epsilon,
                    "Episode Length": step
                })

    def save(self, path:str):
        self.model.save(path)