from configs import maze10, maze3, maze5
from agents import Agent, INDEX2DIRECTIONS
from maze import Maze
import matplotlib.pyplot as plt
from datetime import datetime
import time
import numpy as np
import os
from math import log10

# ------------------------- #
# configs
#  # ## MAZE ## #  #
mazes = {
    "3": maze3,
    "5": maze5,
    "10": maze10,
}
# maze_for_experiment = "10"
#  # ## reward ## #  #
# reward_mode = "sparse"
# reward_mode = "standard"
# reward_scale = 1.0
# eps_greedy_off = False
epsilon = 0.7
#  # ## episode ## #  #
num_episodes = 60
save_freq = 3
# reset or not(if reset is int, reset environment after reset steps)
# reset = float("inf")
reset = 1e5

#  # ## recording ## #  #

yticks = np.arange(0, 1.0, 0.2)

#  # ## minimus steps ## #  #
min_steps = {"3": 2, "5": 8, "10": 16}
# ------------------------- #

maze_sizes = mazes.keys()
reward_modes = ("standard", "sparse")
reward_scales = (1.0, 2.0)
eps_greedy_modes = (True, False)


def experiment(
    maze_for_experiment,
    reward_mode,
    reward_scale,
    eps_greedy_off,
):
    conditions = f"""
    maze: {maze_for_experiment}
    --------------
    reward_mode: {reward_mode}
    reward_scale: {reward_scale}
    epsilon_greedy_off: {eps_greedy_off}
    epsilon: {epsilon}
    --------------
    num_episodes: {num_episodes}
    reset: {reset}
    """
    print(conditions)
    folder = "./results/{}".format(datetime.now().strftime("%m%d-%H%M%S-%f"))
    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(folder + "/conditions.txt", mode="w") as f:
        f.write(conditions)

    env = Maze(maze=mazes[maze_for_experiment], reward_mode=reward_mode)
    plt.imshow(np.logical_not(env.maze).astype(int), cmap=plt.cm.gray_r)
    plt.title("MAZE")
    plt.savefig(folder + "/maze.png")
    plt.clf()

    agent = Agent(env)
    rewards = []
    num_steps = []
    for episode in range(1, num_episodes + 1):
        rewards_in_episode = []
        goaled = False
        state = tuple(env.current_position)
        action = agent.act(
            state, episode, eps_greedy_off=eps_greedy_off, epsilon=epsilon
        )
        steps = 0

        # train loop(start)
        start = time.perf_counter()
        while not goaled:
            reward, goaled = env.step(INDEX2DIRECTIONS[action])
            reward *= reward_scale
            rewards_in_episode.append(reward)

            next_state = tuple(env.current_position)
            next_action = agent.act(
                state, episode, eps_greedy_off=eps_greedy_off, epsilon=epsilon
            )

            agent.update_qtable(state, action, reward, next_state, next_action)

            state, action = next_state, next_action

            steps += 1
            if steps >= reset:
                break
            # elif time.perf_counter() - start > 10:  # 10秒たってもクリアできない場合
            #     break
        # train loop(end)
        env.reset()
        num_steps.append(len(rewards_in_episode))

        # saving records
        if (episode % save_freq) == 0:
            rewards.append(rewards_in_episode)
            plt.plot(range(len(rewards_in_episode)), rewards_in_episode)
            # plt.xticks(
            #     np.arange(0, len(rewards_in_episode), int(log10(len(rewards_in_episode))))
            # )
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

    # testing
    goaled = False
    path = np.zeros_like(env.maze)
    state = tuple(env.current_position)
    path[state] = 1
    action = agent.act_greedy(state)
    step = 0
    not_found = ""
    env.reset()
    while not goaled:
        reward, goaled = env.step(INDEX2DIRECTIONS[action])
        state = tuple(env.current_position)
        action = agent.act_greedy(state)
        path[state] = 1.0
        step += 1
        if step >= reset:
            not_found = "(not found)"
            break
    plt.imshow(path, cmap=plt.cm.gray_r)
    plt.title(f"The path found by the agent {not_found}")
    plt.savefig(folder + "/path_found.png")
    plt.clf()

    xvalues = list(range(len(num_steps)))
    plt.plot(xvalues, num_steps)  # エピソード終了までのステップ数の推移
    minline = list((min_steps[maze_for_experiment] for _ in num_steps))
    plt.plot(
        xvalues,
        minline,
    )
    plt.title("Number of steps until an episode ends")
    plt.savefig(folder + "/num_steps.png")
    plt.clf()
    return rewards, num_steps


# 実験
for maze in maze_sizes:
    for reward_mode in reward_modes:
        for reward_scale in reward_scales:
            for eps_greedy_off in eps_greedy_modes:
                experiment(maze, reward_mode, reward_scale, eps_greedy_off)
