import pickle
from collections import deque

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

from snake_env import Snake
from utils import seed_everything, Experience, ReplayBuffer
from model.dqn_engineered import DQN

env = Snake(env_info={'state_space': "both"})
state = env.reset()

# setting up params
lr = 0.0001
epsilon = 1.0
epsilon_decay = 0.995
gamma = 0.85
training_episodes = 1000

# create new deep q-network instance
model = DQN(env, ((80,80,3),12), lr, gamma, epsilon, epsilon_decay, target_update_interval=5, log_wandb=False)
print(model.model.summary())
model.train(training_episodes, mean_stopping=True)