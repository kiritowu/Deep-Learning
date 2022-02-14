from typing import List, Optional, Tuple

import gym
import tensorflow as tf
import numpy as np
import wandb

from model.dqn import DQN
from utils import ReplayBuffer, Experience_SARSA


class SARSA(DQN):
    def __init__(
        self,
        env: gym.Env,
        observation_space:Tuple[int,int,int],
        lr: float,
        gamma: float,
        epsilon: float,
        epsilon_decay: float,
        target_update_interval: int = 100,
        log_wandb: bool = False,
        replay_buffer: Optional[ReplayBuffer] = None,
        fc_layers: Optional[List[int]] = None,
        conv_layers:Optional[List[int]]=None
    ):
        super().__init__(env, observation_space, lr, gamma, epsilon, epsilon_decay, target_update_interval, log_wandb, replay_buffer, fc_layers, conv_layers)
        self.log_wandb = log_wandb

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
        # getting the states, actions, rewards, next_state, **next_action** and done_list from the random sample
        (
            states,
            actions,
            rewards,
            next_states,
            next_actions,
            done_list,
        ) = self.buffer.sample_sarsa(self.batch_size)

        # calculate the loss to create a target vector for the model to fit with the states
        # On-policy action is choosen replacing the maximum-q value from DQN

        # Stupid way to do indices filtering. Lmk if there are better way to do so
        next_actions_mask = np.tile(np.arange(self.num_action_space), (self.batch_size, 1)) \
            == np.tile(next_actions.reshape(self.batch_size,1), (1,self.num_action_space))
        targets = rewards + self.gamma * np.max(
            self.model.predict_on_batch(next_states) * next_actions_mask, axis=1) \
            * (1 - done_list)
        target_vec = self.model.predict_on_batch(states)
        indexes = np.array([i for i in range(self.batch_size)])
        target_vec[[indexes], [actions]] = targets

        # fit the model with the states and the target vector for one iteration
        self.model.fit(states, target_vec, epochs=1, verbose=0)

    def train(self, num_episodes=1000, max_num_steps = 5000, mean_stopping=True):

        # iterate over the number of episodes
        for episode in range(num_episodes):
            
            state = self.env.reset()
            reward_for_episode = 0
            
            for step in range(max_num_steps):
                # get the action for the current state
                action = self.get_action(state)
                if isinstance(action, tf.Tensor):
                    action = action.numpy()
                
                # get the next_state, reward, done and info after running the action
                next_state, reward, done, info = self.env.step(int(action))
                next_action = self.get_action(next_state) # Get next action for SARSA
                if isinstance(next_action, tf.Tensor):
                    next_action = next_action.numpy()

                # store the experience in replay memory
                self.buffer.append(Experience_SARSA(state, int(action), reward, next_state, int(next_action), done))
                
                # add up rewards
                reward_for_episode += reward
                state = next_state
                
                # train dqn
                self.update_weights()

                if done: break
            
            self.rewards_list.append(reward_for_episode)

            # decay the epsilon after each episode
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            # check for terminal condition
            last_rewards_mean = np.mean(self.rewards_list[-100:])
            if last_rewards_mean > 280 and mean_stopping:
                print("SARSA Training Complete...")
                break
            
            print("[{:0>3}] Reward: {: >8.3f} | Avg Reward: {: >8.3f} | e: {:.3f} | Episode Length: {:>4}"
                  .format(episode, reward_for_episode, last_rewards_mean, self.epsilon, step))

            if self.log_wandb:
                wandb.log({
                    "Episode": episode,
                    "Reward": reward_for_episode,
                    "Avg-Reward-100e": last_rewards_mean,
                    "Epsilon": self.epsilon,
                    "Episode Length": step
                })