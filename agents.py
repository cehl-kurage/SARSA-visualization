# SARSAを実装しましたが、実装が誤っているか、Qtableの更新が上手くいっていないかで学習が上手くいっていません。
import random
from time import sleep

import numpy as np

from maze import Direction, Maze

DIRECTIONS = (Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT)
INDEX2DIRECTIONS = {i: value for i, value in enumerate(DIRECTIONS)}


class Agent:
    def __init__(
        self,
        env: Maze,
        alpha: float = 0.5,
        gamma: float = 0.99,
        max_initial_q: float = 0.1,
    ):
        """Agent for exploring maze environment.

        Args:
            env (Maze): Maze environment.
            alpha (float, optional): . Defaults to 0.5.
            gamma (float, optional): _description_. Defaults to 0.99.
            max_initial_q (float, optional): _description_. Defaults to 0.1.
        """
        self.alpha = alpha
        self.gamma = gamma
        self.qtable = np.random.uniform(
            low=-max_initial_q, high=-max_initial_q, size=(*env.maze.shape, 4)
        )

    def update_qtable(self, state, action, reward, next_state, next_action):
        current_q_index = (*state, action)
        next_q_index = (*next_state, next_action)
        self.qtable[current_q_index] += self.alpha * (
            reward
            + self.gamma * self.qtable[next_q_index]
            - self.qtable[current_q_index]
        )

    def act_randomly(self):
        return random.randint(0, len(DIRECTIONS) - 1)

    def act_greedy(self, state):
        pos = self.qtable[state]
        action = np.argmax(pos)
        return action

    def act(self, state, episode):
        epsilon = 0.7 * (1 / (episode + 1))
        if epsilon < np.random.uniform(0, 1):
            action = self.act_greedy(state)
        else:
            action = self.act_randomly()
        return action


def train(agent: Agent, env: Maze, num_episodes: int = 200, num_steps: int = 30):
    for episode in range(num_episodes):
        state = tuple(env.current_position)
        action = agent.act(state, episode)
        episode_reward = 0

        for t in range(num_steps):
            action = agent.act(state, episode)
            reward, goaled = env.step(INDEX2DIRECTIONS[action])
            next_state = tuple(env.current_position)

            episode_reward += reward

            next_action = agent.act(next_state, episode)

            agent.update_qtable(state, action, reward, next_state, next_action)

            state, action = next_state, next_action
            if goaled:
                break
        env.reset()
        if episode % 50 == 0:
            print(f"Episode {episode} finished / Episode reward: {episode_reward}")


if __name__ == "__main__":
    env = Maze()
    agent = Agent(env)
    train(agent, env, num_episodes=50, num_steps=100)

    # If the qtable does not display well, comment out the part that prints the per-episode reward when learning.
    # print(agent.qtable)

    # input()  # Stop executing to watch q-table.
    goaled = False
    env.reset()
    while not goaled:  # Test the result of training.
        print(env, "\n")
        state = tuple(env.current_position)
        action = agent.act_greedy(state)
        _, goaled = env.step(INDEX2DIRECTIONS[action])
        sleep(0.5)
    print("GOAL")
