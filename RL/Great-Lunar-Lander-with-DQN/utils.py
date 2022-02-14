import collections
import random
from os import stat
from typing import List, Tuple, Union
from datetime import datetime as dt
import wandb

import gym
import pickle
import numpy as np
import tensorflow as tf
from gym.wrappers import Monitor
from tensorflow.keras import backend as K
from tensorflow.keras import initializers
from tensorflow.keras.layers import Dense, InputSpec

from tensorflow.keras.models import load_model


def seed_everything(seed: int = 42) -> None:
    """
    Utility function to seed everything.
    """
    random.seed(1)
    np.random.seed(seed)
    tf.random.set_seed(seed)


Experience = collections.namedtuple(
    "Experience", field_names=["state", "action", "reward", "next_state", "done"]
)

Experience_SARSA = collections.namedtuple(
    "Experience",
    field_names=["state", "action", "reward", "next_state", "next_action", "done"],
)


class ReplayBuffer:
    """
    Replay Buffer for storing past experiences allowing the agent to learn from them
    Args:
        maxlen: maximum size of the buffer

    Modified From: https://gist.github.com/djbyrne/45b6cbc620c8acbef259c7d519bca80f
    """

    def __init__(self, maxlen: int = 1_000_000) -> None:
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
            return (
                np.squeeze(states),
                np.array(actions),
                np.array(rewards),
                np.squeeze(next_states),
                np.array(dones, dtype=np.bool),
            )
        else:
            return (
                np.array(states, dtype=np.float32).squeeze(),
                np.array(actions, dtype=np.float32),
                np.array(rewards, dtype=np.float32),
                np.array(next_states, dtype=np.float32).squeeze(),
                np.array(dones, dtype=np.float32),
            )

    def sample_sarsa(self, batch_size: int) -> Tuple:
        random_sample = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, next_actions, dones = zip(*random_sample)

        return (
            np.array(states).squeeze(),
            np.array(actions),
            np.array(rewards),
            np.array(next_states).squeeze(),
            np.array(next_actions),
            np.array(dones, dtype=np.bool),
        )


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


class NoisyDense(Dense):
    def __init__(self, units, **kwargs):
        self.output_dim = units
        super().__init__(units, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.input_dim = input_shape[-1]

        self.kernel = self.add_weight(
            shape=(self.input_dim, self.units),
            initializer=self.kernel_initializer,
            name="kernel",
            regularizer=None,
            constraint=None,
        )

        self.kernel_sigma = self.add_weight(
            shape=(self.input_dim, self.units),
            initializer=initializers.Constant(0.017),
            name="sigma_kernel",
            regularizer=None,
            constraint=None,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                name="bias",
                regularizer=None,
                constraint=None,
            )
            self.bias_sigma = self.add_weight(
                shape=(self.units,),
                initializer=initializers.Constant(0.017),
                name="bias_sigma",
                regularizer=None,
                constraint=None,
            )
        else:
            self.bias = None

        self.input_spec = InputSpec(min_ndim=2, axes={-1: self.input_dim})
        self.built = True

    def call(self, inputs):
        self.kernel_epsilon = K.random_normal(shape=(self.input_dim, self.units))

        w = self.kernel + tf.multiply(self.kernel_sigma, self.kernel_epsilon)
        output = K.dot(inputs, w)

        if self.use_bias:
            self.bias_epsilon = K.random_normal(shape=(self.units,))

            b = self.bias + tf.multiply(self.bias_sigma, self.bias_epsilon)
            output = output + b
        if self.activation is not None:
            output = self.activation(output)
        return output


# Function to evaluate a trained model
def evaluate_model(
    model_name: str, # The name of te model (e.g. "dqn-final")
    model_path: str, # The path to the model (e.g. "./saved-models/dqn-final.h5")
    num_test_eps: int = 1000,  # The number of test episodes per model
    save_video: bool = False,  # Whether the video of the evaluation is saved
    vid_save_freq: int = 25,  # A video will be saved every <vid_save_freq> episodes
    video_folder: str = "./eval-video",  # The folder to save the video
    log_wandb: bool = False # Whether to log the metrics in Weights and Biases
):

    """
    Returns a dictionary of model names and their reward list & mean reward
    """

    print(
        f"Evaluation Config: \
            \n- Model name: {model_name} \
            \n- Model path: {model_path} \
            \n- No of test episodes/model: {num_test_eps} \
            \n- Save video: {save_video} \
            \n- Video save frequency: {vid_save_freq} \
            \n- Video folder: {video_folder}"
    )

    if log_wandb:
        wandb.config.update({
            "model_name": model_name,
            "model_path": model_path,
            "num_test_eps": num_test_eps,
            "save_video": save_video,
            "vid_save_freq": vid_save_freq,
            "video_folder": video_folder
        })

    # Load the model
    model = load_model(model_path)
    
    metrics_dict = {
        "episode": [],
        "rewards": [],
        "mean_reward": 0,
        "episode_length": [],
        "mean_episode_length": 0,
        "times_landed": 0,
        "percent_landed": 0
    }

    max_num_steps = 1000

    print(f"\nEvaluating {model_name}...")

    env = gym.make("LunarLander-v2")
    env.seed(1)

    if save_video:
        env = Monitor(
            env=env,
            directory=f"{video_folder}/{model_name}",
            video_callable=lambda eps: eps % vid_save_freq == 0,
            force=True,
        )

    # Training Loop
    for test_episode in range(num_test_eps):
        
        current_state = env.reset()
        num_observation_space = env.observation_space.shape[0]
        current_state = np.reshape(current_state, [1, num_observation_space])
        reward_for_episode = 0
        
        for step in range(max_num_steps):
        
            selected_action = np.argmax(model.predict(current_state)[0])
            new_state, reward, done, info = env.step(selected_action)
            new_state = np.reshape(new_state, [1, num_observation_space])
            current_state = new_state
            reward_for_episode += reward

            if done:
                break

        # Record Rewards
        model_reward = metrics_dict["rewards"]
        model_reward.append(reward_for_episode)
        model_mean_reward = sum(model_reward) / len(model_reward)
        metrics_dict["mean_reward"] = model_mean_reward

        # Record Episode Length
        model_episode_length = metrics_dict["episode_length"]
        model_episode_length.append(step+1)
        model_mean_episode_length = sum(model_episode_length) / len(model_episode_length)
        metrics_dict["mean_episode_length"] = model_mean_episode_length

        # Record if it lands
        metrics_dict["times_landed"] += int(reward >= 100)
        metrics_dict["percent_landed"] = metrics_dict["times_landed"] / num_test_eps

        # Record episode number
        metrics_dict["episode"].append(test_episode+1)

        # Log to Weights & Biases
        if log_wandb:
            wandb.log({
                "Episode": test_episode + 1,
                "Reward": reward_for_episode,
                "Mean Reward": model_mean_reward,
                "Episode Length": step+1,
                "Mean Episode Length": model_mean_episode_length,
                "Times Landed": metrics_dict["times_landed"],
                "Percent Landed": metrics_dict["percent_landed"]
            })

        print("{} | [{:0>3}] | R: {:.3f} | Mean(R): {:.3f} | EpsL: {} | Mean(EpsL): {:.0f} | Land: {} | Count(Land): {} | Percent(Land): {:.2f}%".format(
            model_name, test_episode+1, reward_for_episode, model_mean_reward,
            step+1, model_mean_episode_length, reward >= 100,
            metrics_dict["times_landed"], (metrics_dict["times_landed"]/(test_episode+1)) * 100
        ))

    if log_wandb:
        wandb.finish()

    # Save the dictionary after evaluation
    with open(f"./saved-variables/eval_metrics_dict_{model_name.lower()}_{dt.now().strftime('%Y%m%d_%H%M%S')}.p", "wb") as f:
        pickle.dump(metrics_dict, f)

    print(f"Metrics of {model_name} saved as eval_metrics_dict_{model_name.lower()}_{dt.now().strftime('%Y%m%d_%H%M%S')}.p")

    return metrics_dict
