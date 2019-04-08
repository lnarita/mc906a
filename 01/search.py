#!/usr/bin/env python
# coding: utf-8

# Needed to hide warnings in the matplotlib sections
import warnings
from enum import Enum, IntEnum, auto

import matplotlib.cm as cm
# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import time

from maze_generator import *
from search import *

warnings.filterwarnings("ignore")

MASK = int('100000000000000000000000000000000000000000000000000000000000000', 2)


def pluck(dict, *args):
    return (dict[arg] for arg in args)


def elapsed(f):
    start = time.time()
    result = f()
    end = time.time()
    return end - start, result


class Map(dict):
    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v
        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Map, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Map, self).__delitem__(key)
        del self.__dict__[key]


class State(Map):
    __slots__ = ("x", "y", "angle", "depth", "already_faced")

    def __init__(self, x, y, angle, depth, already_faced):
        super().__init__()
        self.x = x
        self.y = y
        self.angle = angle
        self.depth = depth
        self.already_faced = already_faced

    def __eq__(self, other):
        return isinstance(other, State) and self.x == other.x and self.y == other.y

    def __lt__(self, other):
        return self.x < other.x and self.y < other.y

    def __hash__(self):
        return hash(frozenset(self))


class RobotCommons:
    def __init__(self):
        pass

    class Tile(IntEnum):
        GOAL = 0
        START = 1
        EMPTY = 2
        WALL = 3
        VISITED = 4
        SOLUTION = 5
        CURRENT = 6
        NEXT = 7

    class Actions(Enum):
        MV_UP = auto()
        MV_RIGHT = auto()
        MV_LEFT = auto()
        MV_DOWN = auto()
        MV_UP_LEFT = auto()
        MV_UP_RIGHT = auto()
        MV_DOWN_LEFT = auto()
        MV_DOWN_RIGHT = auto()

    class ActionsTheta(Enum):
        ROTATE_LEFT_AND_MOVE = auto()
        ROTATE_RIGHT_AND_MOVE = auto()

    class BlindSearchMethods(Enum):
        BFS = (breadth_first_tree_search,)
        DFS = (depth_first_tree_search,)
        DLS = (depth_limited_search,)
        IDDFS = (iterative_deepening_search,)

        def __init__(self, method):
            self.method = method

    color_map = {
        Tile.EMPTY: [255, 255, 255, 0],
        Tile.WALL: [65, 63, 62, 255],
        Tile.SOLUTION: [24, 144, 136, 255],
        Tile.START: [117, 151, 143, 255],
        Tile.VISITED: [209, 187, 161, 150],
        Tile.GOAL: [235, 101, 89, 255],
        Tile.CURRENT: [184, 15, 10, 175],
        Tile.NEXT: [114, 133, 165, 175]
    }


def fix_tile_map_after_solution(tiles, initial, goal, solution, current=None, next=None):
    result = tiles.copy()
    if solution:
        pos = [(n.state.x, n.state.y) for n in solution.path()]
        rows, cols = np.transpose(pos)
        result[rows, cols] = RobotCommons.Tile.SOLUTION
    result[initial.x, initial.y] = RobotCommons.Tile.START
    result[goal.x, goal.y] = RobotCommons.Tile.GOAL
    if current:
        result[current.x, current.y] = RobotCommons.Tile.CURRENT
    if next:
        result[next.x, next.y] = RobotCommons.Tile.NEXT
    return result


def plot_tile_map(tiles, show_img=True):
    ######## THIS IS FOR IMSHOW ######################################
    width, height = np.shape(tiles)
    data = tiles.astype(int)
    c = np.array([[RobotCommons.color_map.get(v, RobotCommons.color_map.get(RobotCommons.Tile.VISITED)) for v in row] for row in data], dtype='B')

    ######## THIS IS FOR SCATTER ######################################
    # This is for having the coordinates of the scattered points, note that rows' indices
    # map to y coordinates and columns' map to x coordinates
    y, x = np.array([(i, j) for i in range(width) for j in range(height)]).T
    # Scatter does not expect a 3D array of uints but a 2D array of RGB(A) floats
    c1 = (c / 255.0).reshape(width * height, 4)

    ######## THIS IS FOR CMAP ##########################################
    cmap = cm.get_cmap('viridis', len(RobotCommons.color_map))
    aaaaaa = np.array([tuple(np.array(RobotCommons.color_map[b]) / 255.) for b in RobotCommons.Tile], np.dtype('float,float,float,float'))
    cmap.colors = aaaaaa

    ######## THIS IS FOR PLOTTING ######################################
    # two subplots, plot immediately the imshow
    f, ax1 = plt.subplots(nrows=1)
    a1 = ax1.imshow(c, cmap=cmap)
    # Major ticks
    ax1.set_xticks(np.arange(width))
    ax1.set_yticks(np.arange(height))

    # Labels for major ticks
    ax1.set_xticklabels(np.arange(width))
    ax1.set_yticklabels(np.arange(height))

    # Minor ticks
    ax1.set_xticks(np.arange(width) - .5, minor=True)
    ax1.set_yticks(np.arange(height) - .5, minor=True)

    # Gridlines based on minor ticks
    line_color = (90 / 255., 90 / 255., 90 / 255.)
    ax1.grid(which='minor', color=line_color, linestyle='--', linewidth=1)
    # to make a side by side comparison we set the boundaries and aspect
    # of the second plot to mimic imshow's
    # ax2.set_ylim(ax1.get_ylim())
    # ax2.set_xlim(ax1.get_xlim())
    # ax2.set_aspect(1)

    # and finally plot the data
    # sc = ax2.scatter(x, y, c=c1, cmap=cmap, s=70, marker='x')

    cb = f.colorbar(a1, ax=ax1)
    # cb = plt.colorbar(sc, ax=ax2)
    # plt.bar(range(len(y)), y, color = c1)
    cb.set_ticks(np.arange(len(RobotCommons.color_map)) * 36)
    cb.set_ticklabels(['GOAL', 'START', 'EMPTY', 'WALL', 'VISITED', 'SOLUTION', 'CURRENT', 'NEXT'])
    # sc = plt.scatter(x, y, c=c1, s=70, marker='X')
    # set ticks for all subplots
    plt.setp((ax1,), xticks=np.arange(width), yticks=np.arange(height))
    if show_img:
        plt.show()


class Robot(Problem):

    def __init__(self, tiles, alg, initial, goal=None):
        super(Robot, self).__init__(initial, goal)
        self.alg = alg
        self.visited_tiles = tiles.copy()
        self.step = 0

    def actions(self, state):
        x, y, depth = pluck(state, 'x', 'y', 'depth')
        t = self.visited_tiles
        max_x, max_y = np.shape(t)
        left, right, up, down = x - 1, x + 1, y - 1, y + 1
        bit_check = MASK >> state.depth

        def __can_go_up_left():
            return up > 0 and left > 0 and t[left, up] != RobotCommons.Tile.WALL and t[x, up] != RobotCommons.Tile.WALL and t[
                left, y] != RobotCommons.Tile.WALL and (int(t[left, up]) & bit_check) == 0

        def __can_go_down_left():
            return down < max_y and left > 0 and t[left, down] != RobotCommons.Tile.WALL and t[x, down] != RobotCommons.Tile.WALL and t[
                left, y] != RobotCommons.Tile.WALL and (int(t[left, down]) & bit_check) == 0

        def __can_go_up_right():
            return up > 0 and right < max_x and t[right, up] != RobotCommons.Tile.WALL and t[x, up] != RobotCommons.Tile.WALL and t[
                right, y] != RobotCommons.Tile.WALL and (int(t[right, up]) & bit_check) == 0

        def __can_go_down_right():
            return down < max_y and right < max_x and t[right, down] != RobotCommons.Tile.WALL and t[x, down] != RobotCommons.Tile.WALL and t[
                right, y] != RobotCommons.Tile.WALL and (int(t[right, down]) & bit_check) == 0

        def __can_go_up():
            return up > 0 and t[x, up] != RobotCommons.Tile.WALL and (int(t[x, up]) & bit_check) == 0

        def __can_go_down():
            return down < max_y and t[x, down] != RobotCommons.Tile.WALL and (int(t[x, down]) & bit_check) == 0

        def __can_go_left():
            return left > 0 and t[left, y] != RobotCommons.Tile.WALL and (int(t[left, y]) & bit_check) == 0

        def __can_go_right():
            return right < max_x and t[right, y] != RobotCommons.Tile.WALL and (int(t[right, y]) & bit_check) == 0

        if self.alg == RobotCommons.BlindSearchMethods.BFS:
            if __can_go_down_left():
                yield RobotCommons.Actions.MV_DOWN_LEFT

            if __can_go_down_right():
                yield RobotCommons.Actions.MV_DOWN_RIGHT

            if __can_go_up_left():
                yield RobotCommons.Actions.MV_UP_LEFT

            if __can_go_up_right():
                yield RobotCommons.Actions.MV_UP_RIGHT

            if __can_go_up():
                yield RobotCommons.Actions.MV_UP

            if __can_go_down():
                yield RobotCommons.Actions.MV_DOWN

            if __can_go_right():
                yield RobotCommons.Actions.MV_RIGHT

            if __can_go_left():
                yield RobotCommons.Actions.MV_LEFT
        else:
            if __can_go_up():
                yield RobotCommons.Actions.MV_UP

            if __can_go_down():
                yield RobotCommons.Actions.MV_DOWN

            if __can_go_right():
                yield RobotCommons.Actions.MV_RIGHT

            if __can_go_left():
                yield RobotCommons.Actions.MV_LEFT

            if __can_go_up_left():
                yield RobotCommons.Actions.MV_UP_LEFT

            if __can_go_up_right():
                yield RobotCommons.Actions.MV_UP_RIGHT

            if __can_go_down_left():
                yield RobotCommons.Actions.MV_DOWN_LEFT

            if __can_go_down_right():
                yield RobotCommons.Actions.MV_DOWN_RIGHT

    def __mark_visited(self, state):
        if int(self.visited_tiles[state.x, state.y]) & MASK == 0:
            self.visited_tiles[state.x, state.y] = MASK
        bit_check = MASK >> state.depth
        self.visited_tiles[state.x, state.y] = int(self.visited_tiles[state.x, state.y]) | bit_check

    def __move_up(self, s):
        self.__mark_visited(s)
        return State(s.x, s.y - 1, s.angle, s.depth, s.already_faced)

    def __move_down(self, s):
        self.__mark_visited(s)
        return State(s.x, s.y + 1, s.angle, s.depth, s.already_faced)

    def __move_left(self, s):
        self.__mark_visited(s)
        return State(s.x - 1, s.y, s.angle, s.depth, s.already_faced)

    def __move_right(self, s):
        self.__mark_visited(s)
        return State(s.x + 1, s.y, s.angle, s.depth, s.already_faced)

    def __move_up_left(self, s):
        self.__mark_visited(s)
        return State(s.x - 1, s.y - 1, s.angle, s.depth, s.already_faced)

    def __move_down_left(self, s):
        self.__mark_visited(s)
        return State(s.x - 1, s.y + 1, s.angle, s.depth, s.already_faced)

    def __move_up_right(self, s):
        self.__mark_visited(s)
        return State(s.x + 1, s.y - 1, s.angle, s.depth, s.already_faced)

    def __move_down_right(self, s):
        self.__mark_visited(s)
        return State(s.x + 1, s.y + 1, s.angle, s.depth, s.already_faced)

    def result(self, state, action):
        self.step += 1
        func, args = {
            RobotCommons.Actions.MV_UP: (self.__move_up, (state,)),
            RobotCommons.Actions.MV_DOWN: (self.__move_down, (state,)),
            RobotCommons.Actions.MV_LEFT: (self.__move_left, (state,)),
            RobotCommons.Actions.MV_RIGHT: (self.__move_right, (state,)),
            RobotCommons.Actions.MV_UP_RIGHT: (self.__move_up_right, (state,)),
            RobotCommons.Actions.MV_DOWN_RIGHT: (self.__move_down_right, (state,)),
            RobotCommons.Actions.MV_UP_LEFT: (self.__move_up_left, (state,)),
            RobotCommons.Actions.MV_DOWN_LEFT: (self.__move_down_left, (state,)),
        }.get(action)

        next = func(*args)
        plot_tile_map(fix_tile_map_after_solution(self.visited_tiles, self.initial, self.goal, None, state, next), False)
        plt.savefig('./img/{}/{}__{:04d}.png'.format(self.alg, self.alg, self.step))
        plt.cla()
        return next

    def value(self, state):
        pass

    def h(self, node):
        return self.euclidean_distance(node)

    def manhattan_distance(self, node):
        state = node.state
        return abs(state.x - self.goal.x) + abs(state.y - self.goal.y)

    def euclidean_distance(self, node):
        state = node.state
        return int((state.x - self.goal.x) ** 2 + (state.y - self.goal.y) ** 2)


class RobotTheta(Problem):
    class Directions(Enum):
        UP = (90, (RobotCommons.ActionsTheta.ROTATE_LEFT_AND_MOVE, 90))
        UP_RIGHT = (45, (RobotCommons.ActionsTheta.ROTATE_LEFT_AND_MOVE, 45))
        RIGHT = (0, (RobotCommons.ActionsTheta.ROTATE_RIGHT_AND_MOVE, 0))
        DOWN_RIGHT = (315, (RobotCommons.ActionsTheta.ROTATE_RIGHT_AND_MOVE, 45))
        DOWN = (270, (RobotCommons.ActionsTheta.ROTATE_RIGHT_AND_MOVE, 90))
        DOWN_LEFT = (225, (RobotCommons.ActionsTheta.ROTATE_RIGHT_AND_MOVE, 135))
        LEFT = (180, (RobotCommons.ActionsTheta.ROTATE_LEFT_AND_MOVE, 180))
        UP_LEFT = (135, (RobotCommons.ActionsTheta.ROTATE_LEFT_AND_MOVE, 135))

        def __init__(self, angle, action):
            self.angle = angle
            self.action = action

    def __init__(self, alg, initial, goal=None):
        super(RobotTheta, self).__init__(initial, goal)
        self.alg = alg
        self.tiles = initial.tiles

    def actions(self, state):
        x, y, a, _, already_faced = state
        max_x, max_y = np.shape(self.tiles)
        left, right, up, down = x - 1, x + 1, y - 1, y + 1
        t = self.tiles

        def __can_go_up_left():
            return 0 < up < max_y and 0 < left < max_x and t[left, up] != RobotCommons.Tile.WALL and t[
                left, up] != RobotCommons.Tile.VISITED and RobotTheta.Directions.UP_LEFT.angle not in already_faced

        def __can_go_down_left():
            return 0 < down < max_y and 0 < left < max_x and t[left, down] != RobotCommons.Tile.WALL and t[
                left, down] != RobotCommons.Tile.VISITED and RobotTheta.Directions.DOWN_LEFT.angle not in already_faced

        def __can_go_up_right():
            return 0 < up < max_y and 0 < right < max_x and t[right, up] != RobotCommons.Tile.WALL and t[
                right, up] != RobotCommons.Tile.VISITED and RobotTheta.Directions.UP_RIGHT.angle not in already_faced

        def __can_go_down_right():
            return 0 < down < max_x and 0 < right < max_x and t[right, down] != RobotCommons.Tile.WALL and t[
                right, down] != RobotCommons.Tile.VISITED and RobotTheta.Directions.DOWN_RIGHT.angle not in already_faced

        def __can_go_up():
            return 0 < up < max_y and 0 <= x < max_x and t[x, up] != RobotCommons.Tile.WALL and t[
                x, up] != RobotCommons.Tile.VISITED and RobotTheta.Directions.UP.angle not in already_faced

        def __can_go_down():
            return 0 < down < max_y and 0 <= x < max_x and t[x, down] != RobotCommons.Tile.WALL and t[
                x, down] != RobotCommons.Tile.VISITED and RobotTheta.Directions.DOWN.angle not in already_faced

        def __can_go_left():
            return 0 < left < max_x and 0 <= y < max_y and t[left, y] != RobotCommons.Tile.WALL and t[
                left, y] != RobotCommons.Tile.VISITED and 270 not in already_faced

        def __can_go_right():
            return 0 < right < max_x and 0 <= y < max_y and t[right, y] != RobotCommons.Tile.WALL and t[
                right, y] != RobotCommons.Tile.VISITED and RobotTheta.Directions.RIGHT.angle not in already_faced

        if __can_go_up_left():
            yield (RobotTheta.Directions.UP_LEFT,) + RobotTheta.Directions.UP_LEFT.action

        if __can_go_up_right():
            yield (RobotTheta.Directions.UP_RIGHT,) + RobotTheta.Directions.UP_RIGHT.action

        if __can_go_down_left():
            yield (RobotTheta.Directions.DOWN_LEFT,) + RobotTheta.Directions.DOWN_LEFT.action

        if __can_go_down_right():
            yield (RobotTheta.Directions.DOWN_RIGHT,) + RobotTheta.Directions.DOWN_RIGHT.action

        if __can_go_up():
            yield (RobotTheta.Directions.UP,) + RobotTheta.Directions.UP.action

        if __can_go_down():
            yield (RobotTheta.Directions.DOWN,) + RobotTheta.Directions.DOWN.action

        if __can_go_right():
            yield (RobotTheta.Directions.RIGHT,) + RobotTheta.Directions.RIGHT.action

        if __can_go_left():
            yield (RobotTheta.Directions.LEFT,) + RobotTheta.Directions.LEFT.action

    def __mark_visited(self, state):
        self.tiles[state.x, state.y] = RobotCommons.Tile.VISITED

    def __rotate(self, s, angle):
        new_angle = (s.angle + angle) % 360
        new_state = State(s.x, s.y, new_angle, s.tiles, s.already_faced + [new_angle])
        return new_state

    def __move_up(self, s):
        self.__mark_visited(s)
        return State(s.x, s.y - 1, s.angle, s.tiles, [])

    def __move_down(self, s):
        self.__mark_visited(s)
        return State(s.x, s.y + 1, s.angle, s.tiles, [])

    def __move_left(self, s):
        self.__mark_visited(s)
        return State(s.x - 1, s.y, s.angle, s.tiles, [])

    def __move_right(self, s):
        self.__mark_visited(s)
        return State(s.x + 1, s.y, s.angle, s.tiles, [])

    def __move_up_left(self, s):
        self.__mark_visited(s)
        return State(s.x - 1, s.y - 1, s.angle, s.tiles, [])

    def __move_down_left(self, s):
        self.__mark_visited(s)
        return State(s.x - 1, s.y + 1, s.angle, s.tiles, [])

    def __move_up_right(self, s):
        self.__mark_visited(s)
        return State(s.x + 1, s.y - 1, s.angle, s.tiles, [])

    def __move_down_right(self, s):
        self.__mark_visited(s)
        return State(s.x + 1, s.y + 1, s.angle, s.tiles, [])

    def result(self, state, action):
        debug, a, param = action
        if a == RobotCommons.ActionsTheta.ROTATE_LEFT_AND_MOVE:
            turn = - state.angle
            new_state = self.__rotate(state, turn + param)
        else:
            turn = - state.angle
            new_state = self.__rotate(state, turn + 360 - param)

        self.tiles[state.x, state.y] = RobotCommons.Tile.VISITED

        result = None
        if RobotTheta.Directions.RIGHT.angle <= new_state.angle < RobotTheta.Directions.UP_RIGHT.angle:
            result = self.__move_right(new_state)
        elif RobotTheta.Directions.UP_RIGHT.angle <= new_state.angle < RobotTheta.Directions.UP.angle:
            result = self.__move_up_right(new_state)
        elif RobotTheta.Directions.UP.angle <= new_state.angle < RobotTheta.Directions.UP_LEFT.angle:
            result = self.__move_up(new_state)
        elif RobotTheta.Directions.UP_LEFT.angle <= new_state.angle < RobotTheta.Directions.LEFT.angle:
            result = self.__move_up_left(new_state)
        elif RobotTheta.Directions.LEFT.angle <= new_state.angle < RobotTheta.Directions.DOWN_LEFT.angle:
            result = self.__move_left(new_state)
        elif RobotTheta.Directions.DOWN_LEFT.angle <= new_state.angle < RobotTheta.Directions.DOWN.angle:
            result = self.__move_down_left(new_state)
        elif RobotTheta.Directions.DOWN.angle <= new_state.angle < RobotTheta.Directions.DOWN_RIGHT.angle:
            result = self.__move_down(new_state)
        elif RobotTheta.Directions.DOWN_RIGHT.angle <= new_state.angle < 360:
            result = self.__move_down_right(new_state)

        return result

    # def goal_test(self, state):
    #     (gx, gy, _, _, _) = self.goal
    #     (x, y, _, _, _) = state
    #     return x == gx and y == gy

    def value(self, state):
        pass


def generate_maze(width, height, walls):
    if type(walls) == list:
        tiles = np.ones((width, height)).astype(int) * RobotCommons.Tile.EMPTY
        if walls:
            rows, cols = np.transpose(walls)
            tiles[rows, cols] = RobotCommons.Tile.WALL
    elif type(walls) == int:
        generated_maze = Maze.generate(width, height)
        tiles = []
        for line in generated_maze._to_str_matrix():
            row = []
            for c in line:
                if c == ' ':
                    row.append(RobotCommons.Tile.EMPTY)  # spaces are 1s
                else:
                    row.append(RobotCommons.Tile.WALL)  # walls are 0s
            tiles.append(row)
    else:
        raise TypeError("wall must be either a list or an int")
    return tiles


class RobotProblemFactory:
    def __init__(self, problem, alg, width, height, start, goal, walls=[]):
        self.start = start
        self.goal = goal
        sx, sy, st, depth = pluck(start, 'x', 'y', 'angle', 'depth')
        gx, gy = pluck(goal, 'x', 'y')
        self.tiles = np.array(generate_maze(width, height, walls))
        self.tiles[sx, sy] = RobotCommons.Tile.START
        self.tiles[gx, gy] = RobotCommons.Tile.GOAL
        self.instance = problem(self.tiles.copy(), alg, State(sx, sy, st, depth, []), goal)


if __name__ == '__main__':
    width, height = 10, 10
    start = State(0, 0, 90, 0, None)
    goal = State(9, 9, 0, 0, None)
    walls = [(i, 2) for i in range(9, 2, -1)] + [(i, 7) for i in range(0, 8)] + [(5, 4), (4, 5)]

    # width, height = 10, 10
    # start = State(1, 1, 90, None, None)
    # goal = State(19, 19, 0, None, None)
    # walls = 1

    for alg in RobotCommons.BlindSearchMethods:
        problem = RobotProblemFactory(Robot, alg, width, height, start, goal, walls)
        if alg != RobotCommons.BlindSearchMethods.IDDFS:
            continue
            # plot_tile_map(fix_tile_map_after_solution(problem.tiles, problem.start, problem.goal, None, None, None), False)
            # plt.savefig('./img/{}/{}__{:04d}.png'.format(alg, alg, 0))
            # plt.cla()
            # e, solution = elapsed(lambda: alg.method(problem.instance))
            # print('Robot ', alg, ' found solution in ', e, 's : ', solution.path())
            # plot_tile_map(fix_tile_map_after_solution(problem.instance.visited_tiles, problem.start, problem.goal, solution), False)
            # plt.savefig('./img/{}/{}__{:04d}.png'.format(alg, alg, 6000))
            # plt.cla()
            # print(alg)
        else:
            def __iddfs():
                for depth in range(sys.maxsize):
                    plot_tile_map(fix_tile_map_after_solution(problem.tiles, problem.start, problem.goal, None, None, None), False)
                    plt.savefig('./img/{}/{}__{:04d}.png'.format(RobotCommons.BlindSearchMethods.IDDFS, RobotCommons.BlindSearchMethods.IDDFS, 0))
                    plt.cla()
                    result = depth_limited_search(problem.instance, depth)
                    new_state = problem.instance.initial
                    problem.instance.initial = State(new_state.x, new_state.y, new_state.angle, depth, new_state.already_faced)
                    if result != 'cutoff':
                        return result


            e, solution = elapsed(lambda: __iddfs())
            print('Robot ', alg, ' found solution in ', e, 's : ', solution.path())
            plot_tile_map(fix_tile_map_after_solution(problem.instance.visited_tiles, problem.start, problem.goal, solution), False)
            plt.savefig('./img/{}/{}__{:04d}.png'.format(RobotCommons.BlindSearchMethods.IDDFS, RobotCommons.BlindSearchMethods.IDDFS, 6000))
            plt.cla()

    problem = RobotProblemFactory(Robot, None, width, height, start, goal, walls)
    solution = astar_search(problem.instance, problem.instance.manhattan_distance)
    print('A* using first heuristic: ', solution.path())
    plot_tile_map(fix_tile_map_after_solution(problem.instance.visited_tiles, problem.start, problem.goal, solution))

    problem = RobotProblemFactory(Robot, None, width, height, start, goal, walls)
    solution = astar_search(problem.instance, problem.instance.euclidean_distance)
    print('A* using second heuristic: ', solution.path())
    plot_tile_map(fix_tile_map_after_solution(problem.instance.visited_tiles, problem.start, problem.goal, solution))

    # for alg in RobotCommons.BlindSearchMethods:
    #     if alg != RobotCommons.BlindSearchMethods.IDDFS:
    #         problem = RobotProblemFactory(RobotTheta, alg, width, height, start, goal, walls)
    #         solution = alg.method(problem.instance)
    #         print('Robot ', alg, ': ', solution.path())
    #         plot_tile_map(fix_tile_map_after_solution(problem.tiles, problem.start, problem.goal, solution))

    # for depth in range(0, 50):
    #     problem = RobotProblemFactory(RobotTheta, RobotCommons.BlindSearchMethods.IDDFS, width, height, start, goal, walls)
    #     result = depth_limited_search(problem.instance, depth)
    #     if result != 'cutoff':
    #         print('Robot IDDFS: ', result.path())
    #         plot_tile_map(fix_tile_map_after_solution(problem.tiles, problem.start, problem.goal, result))
    #         break
    # problem = RobotProblemFactory(RobotII, width, height, start, goal, walls)
    # solution = depth_first_tree_search(problem.instance)
    # print('Robot II DFS: ', solution.path())
    #
    # plot_tile_map(fix_tile_map_after_solution(problem.tiles, problem.start, problem.goal, solution))
    #
    # problem = RobotProblemFactory(RobotII, width, height, start, goal, walls)
    # solution = breadth_first_tree_search(problem.instance)
    # print('Robot II BFS : ', solution.path())
    #
    # plot_tile_map(fix_tile_map_after_solution(problem.tiles, problem.start, problem.goal, solution))
    #
    # problem = RobotProblemFactory(RobotII, width, height, start, goal, walls)
    # solution = depth_limited_search(problem.instance)
    # print('Robot II DLS: ', solution.path())
    #
    # plot_tile_map(fix_tile_map_after_solution(problem.tiles, problem.start, problem.goal, solution))
    #
    # problem = RobotProblemFactory(RobotII, width, height, start, goal, walls)
    # for depth in range(0, 50):
    #     result = depth_limited_search(problem.instance, depth)
    #     if result != 'cutoff':
    #         solution = result
    #         break
    #
    # print('RobotII IDFS: ', solution.path())
    #
    # plot_tile_map(fix_tile_map_after_solution(problem.instance.tiles, problem.start, problem.goal, solution))

    # problem = RobotProblemFactory(RobotTheta, width, height, start, goal, walls)
    # solution = breadth_first_tree_search(problem.instance)
    # print('Robot Theta BFS: ', solution.path())
    #
    # plot_tile_map(fix_tile_map_after_solution(problem.tiles, problem.start, problem.goal, solution))

    # problem = RobotProblemFactory(Robot, None, width, height, start, goal, walls)
    # headers = ['Alg.', 'Robot (10, 10)']
    # compare_searchers([problem.instance], headers)
