import collections
import random
from typing import Tuple, Union, List
import numpy as np
import tensorflow as tf
import gym
from gym.wrappers import Monitor
from tensorflow.keras.models import load_model

def seed_everything(seed:int=42)->None:
    """
    Utility function to seed everything.
    """
    random.seed(1)
    np.random.seed(seed)
    tf.random.set_seed(seed)


Experience = collections.namedtuple(
    'Experience', field_names=['state', 'action', 'reward',
                               'next_state', 'done'])

Experience_SARSA = collections.namedtuple(
    'Experience', field_names=['state', 'action', 'reward',
                               'next_state', 'next_action', 'done'])

class ReplayBuffer:
    """
    Replay Buffer for storing past experiences allowing the agent to learn from them
    Args:
        maxlen: maximum size of the buffer
    
    Modified From: https://gist.github.com/djbyrne/45b6cbc620c8acbef259c7d519bca80f
    """

    def __init__(self, maxlen: int=1_000_000) -> None:
        self.buffer = collections.deque(maxlen=maxlen)

    def __len__(self) -> None:
        return len(self.buffer)

    def append(self, experience: Union[Experience, Experience_SARSA]) -> None:
        """
        Add experience to the buffer
        Args:
            experience: tuple (state, action, reward, new_state, *next_action, done)
        
        Remarks:
        *next_action is only available for SARSA.
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int, dqn: bool = False) -> Tuple:
        random_sample = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*random_sample)
        if dqn:
            return [np.array(states)[:,0],np.array(states)[:,1]], np.array(actions), np.array(rewards), [np.array(next_states)[:,0],np.array(next_states)[:,1]], np.array(dones, dtype=np.bool)
        else:
            return np.array(states, dtype=np.float32).squeeze(), np.array(actions, dtype=np.float32), np.array(rewards, dtype=np.float32), \
                   np.array(next_states, dtype=np.float32).squeeze(), np.array(dones, dtype=np.float32)

    def sample_sarsa(self, batch_size: int) -> Tuple:
        random_sample = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, next_actions, dones = zip(*random_sample)

        return np.array(states).squeeze(), np.array(actions), np.array(rewards), \
            np.array(next_states).squeeze(), np.array(next_actions), np.array(dones, dtype=np.bool)


class OUActionNoise:
    """
    Ornstein-Uhlenbeck Noise process adapted from
    https://keras.io/examples/rl/ddpg_pendulum/
    """
    def __init__(self, mean=0, std_deviation=0.2, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

def evaluate_models(
        models_name_path: List[Tuple[str,str]],     # Example: [("dqn", "./relative/path.h5"),...]
        num_test_eps: int = 1000,                   # The number of test episodes per model
        save_video: bool = False,                   # Whether the video of the evaluation is saved
        vid_save_freq: int = 25,                    # A video will be saved every <vid_save_freq> episodes
        video_folder: str = "./evaluation_video",   # The folder to save the video
    ):

    """
    Returns a dictionary of model names and their reward list & mean reward
    """

    print(f"Evaluation Config: \
            \n- No of models: {len(models_name_path)} \
            \n- No of test episodes/model: {num_test_eps} \
            \n- Save video: {save_video} \
            \n- Video save frequency: {vid_save_freq} \
            \n- Video folder: {video_folder}")

    models = [(n,load_model(p)) for n,p in models_name_path]
    rewards_dict = { m[0]:{"rewards":[], "mean_reward":0} for m in models_name_path }

    max_num_steps = 1000

    for model_name, model in models:
        print(f"\nEvaluating {model_name}...")

        env = gym.make("LunarLander-v2")
        env.seed(1)
        
        if save_video:
            env = Monitor(
                env = env,
                directory = f"{video_folder}/{model_name}",
                video_callable = lambda eps: eps % vid_save_freq == 0,
                force = True
            )

        for test_episode in range(num_test_eps):
            current_state = env.reset()
            num_observation_space = env.observation_space.shape[0]
            current_state = np.reshape(current_state, [1, num_observation_space])
            reward_for_episode = 0
            
            for _ in range(max_num_steps):
                selected_action = np.argmax(model.predict(current_state)[0])
                new_state, reward, done, info = env.step(selected_action)
                new_state = np.reshape(new_state, [1, num_observation_space])
                current_state = new_state
                reward_for_episode += reward
            
                if done: break

            model_reward = rewards_dict[model_name]["rewards"]
            model_reward.append(reward_for_episode)
            model_mean_reward = sum(model_reward) / len(model_reward)
            rewards_dict[model_name]["mean_reward"] = model_mean_reward
            
            print(f'{model_name} | [{test_episode + 1:0>2}] Reward: {reward_for_episode:.3f} | Mean Reward: {model_mean_reward:.3f}')

    return rewards_dict