# 仕様？
# 居られる場所、いられない場所をbool値で示す
# 移動は標準入力で行う
# 迷路はnumpyの2d arrayで表現
# 現在地は特殊な表示
# mazeクラス
# up, down, right, leftの各メソッドで動く
# current_position属性を持っておいて、それに対して各操作をするタイミングで正常性を検証する
# 正常性のチェックは動くメソッドの外に実装

from copy import deepcopy

import numpy as np


class Direction:
    UP = "up"
    DOWN = "down"
    RIGHT = "right"
    LEFT = "left"


class Maze:
    def __init__(
        self,
        size=(5, 5),
        wall_index=[2, 9, 11, 12, 14, 15, 22, 24],
        maze=None,
        reward_mode="standard",
    ) -> None:
        if maze is not None:
            self.maze = maze
            walls = np.where(np.logical_not(maze))
            self.walls = [np.array(i, j) for i, j in zip(walls)]
        else:
            flat_maze = np.full(size[0] * size[1], True)
            wall_index = np.array(wall_index) - 1
            flat_maze[[wall_index]] = False
            self.maze = flat_maze.reshape(size)
            self.walls = [np.array([i // size[0], i % size[0]]) for i in wall_index]
        self.commands = (Direction.UP, Direction.DOWN, Direction.RIGHT, Direction.LEFT)
        self.current_position = np.zeros((2,), int)
        self.reward_mode = reward_mode

    def up(self):
        next_position = deepcopy(self.current_position)
        next_position[0] -= 1
        return next_position

    def down(self):
        next_position = deepcopy(self.current_position)  # こうしないと参照渡しになって4にます。
        next_position[0] += 1
        return next_position

    def right(self):
        next_position = deepcopy(self.current_position)
        next_position[1] += 1
        return next_position

    def left(self):
        next_position = deepcopy(self.current_position)
        next_position[1] -= 1
        return next_position

    def move_to(self, command: str) -> bool:
        command = command.lower()
        if command not in self.commands:
            return False, False

        match command:
            case Direction.UP:
                next_position = self.up()
            case Direction.DOWN:
                next_position = self.down()
            case Direction.LEFT:
                next_position = self.left()
            case Direction.RIGHT:
                next_position = self.right()
        is_valid_position = self.is_valid_position(next_position)
        if is_valid_position:
            self.current_position = next_position
        return is_valid_position, self.is_goaled()

    def is_goaled(self):
        return (self.current_position == (np.array(self.maze.shape) - 1)).all()

    def is_valid_position(self, next_position):
        return not (
            (next_position < 0).any()
            or (next_position >= self.maze.shape).any()
            or any([(next_position == wall).all() for wall in self.walls])
        )

    def reset(self):
        self.current_position = np.zeros((2,), int)

    def step(self, action: int):
        """step into time t+1 by applying action to this env.

        Args:
            action (int): Action made by an agent.
        Returns:
            reward (float): Reward.
            goaled (bool): if goaled then True.
        """
        act_info = self.move_to(action)
        reward = self.calc_reward(*act_info)
        return reward, self.is_goaled()

    def calc_reward(self, moved_successfully, is_goaled):
        reward = 0.0
        if self.reward_mode == "standard":
            if not moved_successfully:
                reward -= 0.1
            if is_goaled:
                reward += 1.0
        else:
            pass
        return reward

    def __str__(self):
        current_maze = np.where(self.maze, "□", "■")
        current_maze[tuple(self.current_position)] = "*"
        return str(current_maze)


if __name__ == "__main__":
    maze = Maze()
    goaled = False
    print(maze)
    while not goaled:
        command = input(f"動く方向を{maze.commands}から指定してください: ")
        moved_successfully, goaled = maze.move_to(command)
        print(maze)
    print("!!!!! goal !!!!!")
    print(maze.maze)
