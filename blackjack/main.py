from __future__ import annotations
from collections import defaultdict

import matplotlib.pyplot as plt
# from matplotlib.patches import patch

import numpy as np
import seaborn as sns
from gymnasium import Env
from gymnasium.wrappers import RecordEpisodeStatistics
from tqdm import tqdm
from IPython.display import clear_output
import gymnasium as gym


# reset env
# done = False
# observation, info = env.reset()

# observation = (16,9,False)
# observation = (player current sum,dealer faceup card, player holds usable ace)

# sample actions
# action 1:
# action = env.action_space.sample()

# execute action
# observation, reward, terminated, truncated, info = env.step(action)


class BlackJackAgent:
    def __init__(
            self,
            env: Env,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            discount_factor: float = 0.95
    ):
        self.env = env
        # Q-value used to estimate optimal action to take at each step
        # It is the one that maximises the long term reward and is given by:
        #
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.lr = learning_rate
        # make episode terminate if terminal state is long winded or infinite
        self.discount_factor = discount_factor
        # epsilon is the exploration factor, it give the probability of picking a random action
        # instead of maximizing the reward
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """"
        Return the best action with probability (1-epsilon) otherwise pick a random action
        with probability epsilon
        """
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(self, current_obs: tuple[int, int, bool], current_action: int, current_reward: float,
               is_terminated: bool, next_obs: tuple[int, int, bool]):
        """
            Updates Q-value of actions
        """
        future_q_value = (not is_terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (current_reward
                               + self.discount_factor * future_q_value
                               - self.q_values[current_obs][current_action])
        self.q_values[current_obs][current_action] = (self.q_values[current_obs][current_action]
                                                      + self.lr * temporal_difference)
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


def main():
    env = gym.make('Blackjack-v1', render_mode="rgb_array")
    # hyperparameters
    learning_rate = 0.01
    n_episodes = 1000
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce exploration over time
    final_epsilon = 0.1

    agent = BlackJackAgent(env=env, learning_rate=learning_rate, initial_epsilon=start_epsilon,
                           epsilon_decay=epsilon_decay, final_epsilon=final_epsilon)

    print("doing something...")
    env_wrapper = gym.wrappers.RecordEpisodeStatistics(env=env, deque_size=n_episodes)
    for episode in tqdm(range(n_episodes)):
        obs, current_info = env_wrapper.reset()
        current_done = False
        clear_output()

        # play one episode
        while not current_done:
            current_action = agent.get_action(obs)
            next_obs, next_reward, next_terminated, next_truncated, next_info = env_wrapper.step(
                current_action)

            # print(next_reward)
            agent.update(obs, current_action, float(next_reward), next_terminated, next_obs)
            frame = env_wrapper.render()
            plt.imshow(frame)
            # plt.show()

            current_done = next_terminated or next_truncated
            obs = next_obs

        agent.decay_epsilon()

    plot_metrics(env_wrapper=env_wrapper, agent=agent)


def plot_metrics(env_wrapper: RecordEpisodeStatistics, agent: BlackJackAgent):
    rolling_length = 30
    fig, axs = plt.subplots(ncols=3, figsize=(12,5))
    axs[0].set_title("Episode reward")
    reward_moving_average = (
        np.convolve(np.array(env_wrapper.return_queue).flatten(), np.ones(rolling_length), mode="valid")
    )/rolling_length
    axs[0].plot(range(len(reward_moving_average)),reward_moving_average)

    axs[1].set_title("Episode lengths")

    length_moving_average = (
        np.convolve(np.array(env_wrapper.length_queue).flatten(), np.ones(rolling_length), mode="valid")
    )/rolling_length
    axs[1].plot(range(len(length_moving_average)), length_moving_average)

    training_error_moving_average = (
        np.convolve(agent.training_error, np.ones(rolling_length), mode="valid")
        ) / rolling_length
    axs[2].set_title("Training error")
    axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
    print(training_error_moving_average)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
