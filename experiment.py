from configs import maze10, maze3, maze5
from agents import Agent, INDEX2DIRECTIONS
from maze import Maze
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import os

# ------------------------- #
# configs
#  # ## MAZE ## #  #
mazes = {
    "3": maze3,
    "5": maze5,
    "10": maze10,
}
maze_for_experiment = "3"
#  # ## reward ## #  #
# reward_mode = "sparse"
reward_mode = "standard"
reward_scale = 1.0

#  # ## episode ## #  #
num_episodes = 21
save_freq = 3
# reset or not(if reset is int, reset environment after reset steps)
reset = float("inf")

#  # ## recording ## #  #
folder = "./results/{}".format(datetime.now().strftime("%m%d-%H%M%S"))
yticks = np.arange(0, 1.0, 0.2)
# ------------------------- #



conditions = f"""
maze: {maze_for_experiment}
--------------
reward_mode: {reward_mode}
reward_scale: {reward_scale}
--------------
num_episodes: {num_episodes}
reset: {reset}
"""

if not os.path.exists(folder):
    os.makedirs(folder)

with open(folder + "/conditions.txt", mode="w") as f:
    f.write(conditions)

env = Maze(maze=mazes[maze_for_experiment], reward_mode=reward_mode)
agent = Agent(env)
rewards = []
for episode in range(1, num_episodes + 1):
    rewards_in_episode = []
    goaled = False
    state = tuple(env.current_position)
    action = agent.act(state, episode)
    steps = 0

    # train loop(start)
    while not goaled:
        reward, goaled = env.step(INDEX2DIRECTIONS[action])
        reward *= reward_scale
        rewards_in_episode.append(reward)

        next_state = tuple(env.current_position)
        next_action = agent.act(next_state, episode)

        agent.update_qtable(state, action, reward, next_state, next_action)

        state, action = next_state, next_action

        steps += 1
        if steps >= reset:
            break
    # train loop(end)
    env.reset()

    # saving records
    if (episode % save_freq) == 0:
        rewards.append(rewards_in_episode)
        plt.plot(range(len(rewards_in_episode)), rewards_in_episode)
        plt.xticks(range(len(rewards_in_episode)))
        plt.yticks(yticks)
        plt.title(f"rewards in episode{episode}")
        plt.savefig(folder + f"/reward_in_episode{episode}.png")
        plt.clf()

        qtable = agent.qtable
        vtable = qtable.sum(axis=-1)
        mappable = plt.imshow(vtable, cmap=plt.cm.gray_r)
        plt.colorbar(mappable)
        plt.savefig(folder + f"/value-table_in_episode{episode}.png")
        plt.clf()
