import gym
import numpy as np
import tensorflow as tf
from typing import List, Optional
from utils import ReplayBuffer
from .dqn import DQN


class DoubleDQN(DQN):
    def __init__(
        self,
        env: gym.Env,
        lr: float,
        gamma: float,
        epsilon: float,
        epsilon_decay: float,
        target_update_interval: int = 100,
        log_wandb: bool = False,
        replay_buffer: Optional[ReplayBuffer] = None,
        layers: Optional[List[int]] = None,
    ):
        super().__init__(
            env,
            lr,
            gamma,
            epsilon,
            epsilon_decay,
            target_update_interval,
            replay_buffer,
            layers,
        )
        self.log_wandb = log_wandb

    def update_weights(self):
        # buffer size check
        if len(self.buffer) < self.batch_size:
            return

        # prevent over-training: stop learning if the mean of the latest 10 rewards is greater than 180
        # this causes "RuntimeWarning: Mean of empty slice."
        if np.mean(self.rewards_list[-10:]) > 200:
            return

        # randomly sample a replay memory with the size of the batch
        # getting the states, actions, rewards, next_state and done_list from the random sample
        states, actions, rewards, next_states, done_list = self.buffer.sample(
            self.batch_size
        )

        online_net_selected_actions = np.argmax(
            self.model.predict_on_batch(next_states), axis=1
        )  # 64x1: For each batch, the index of the selected action to take
        # Assuming a batch size of 1, the calculation goes something like (assume action selected is 4):
        # [0 1 2 3 4] == [4 4 4 4] -> [False False False True]
        # print(f"Online Net Selected Actions: {online_net_selected_actions}")
        actions_mask = np.tile(
            np.arange(self.num_action_space), (self.batch_size, 1)
        ) == np.tile(
            online_net_selected_actions.reshape(self.batch_size, 1),
            (1, self.num_action_space),
        )
        target_net_q_values = self.model_target.predict_on_batch(next_states)  # 64x4: q values for each action selected
        # print(f"Target Q values: {target_net_q_values}") 
        target_net_q_values = np.max(
            target_net_q_values * actions_mask, axis=1
        )  # Select the q-values of the selected actions
        # print(f"Target Q values: {target_net_q_values}")
        # # calculate the loss to create a target vector for the model to fit with the states
        targets = rewards + self.gamma * target_net_q_values * (1 - done_list)  # 64x1
        target_vec = self.model.predict_on_batch(states)
        indexes = np.array([i for i in range(self.batch_size)])
        target_vec[[indexes], [actions.astype(np.int64)]] = targets
        # print(f"Target vector: {target_vec}")
        # fit the model with the states and the target vector for one iteration
        self.model.fit(states, target_vec, epochs=1, verbose=0)
