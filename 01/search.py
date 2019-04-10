#!/usr/bin/env python
# coding: utf-8

import time
# Needed to hide warnings in the matplotlib sections
import warnings
from enum import Enum, IntEnum, auto

import matplotlib.cm as cm
# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np

from maze_generator import *
from search import *

LAST_FRAME = sys.maxsize

warnings.filterwarnings("ignore")


def pluck(dict, *args):
    return (dict[arg] for arg in args)


def elapsed(f):
    t_start = time.time()
    result = f()
    t_end = time.time()
    return t_end - t_start, result


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
    __slots__ = ("x", "y", "last_movement", "already_visited")

    def __init__(self, x, y, last_movement=None, already_visited=None):
        super().__init__()
        if already_visited is None:
            already_visited = set()
        self.x = x
        self.y = y
        self.last_movement = last_movement
        self.already_visited = already_visited

    def __eq__(self, other):
        return isinstance(other, State) and self.x == other.x and self.y == other.y

    def __lt__(self, other):
        return self.x < other.x and self.y < other.y

    def __hash__(self):
        return hash(frozenset(self))


class StateBFS(Map):
    __slots__ = ("x", "y", "last_movement")

    def __init__(self, x, y, last_movement=None):
        super().__init__()
        self.x = x
        self.y = y
        self.last_movement = last_movement

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

    class BlindSearchMethods(Enum):
        BFS = (breadth_first_graph_search,)
        DFS = (depth_first_graph_search,)
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

    up = frozenset((Actions.MV_DOWN, Actions.MV_DOWN_RIGHT, Actions.MV_DOWN_LEFT))
    down = frozenset((Actions.MV_UP, Actions.MV_UP_RIGHT, Actions.MV_UP_LEFT))
    right = frozenset((Actions.MV_LEFT, Actions.MV_UP_LEFT, Actions.MV_DOWN_LEFT))
    left = frozenset((Actions.MV_RIGHT, Actions.MV_UP_RIGHT, Actions.MV_DOWN_RIGHT))
    up_left = frozenset((Actions.MV_DOWN, Actions.MV_RIGHT, Actions.MV_DOWN_RIGHT))
    up_right = frozenset((Actions.MV_DOWN, Actions.MV_LEFT, Actions.MV_DOWN_LEFT))
    down_left = frozenset((Actions.MV_UP, Actions.MV_RIGHT, Actions.MV_UP_RIGHT))
    down_right = frozenset((Actions.MV_UP, Actions.MV_LEFT, Actions.MV_UP_LEFT))


def fix_tile_map_after_solution(tiles, initial, goal, solution, current=None, next=None, copy_tiles=True):
    if copy_tiles:
        result = tiles.copy()
    else:
        result = tiles
    if solution:
        pos = [(n.state.x, n.state.y) for n in solution.path()]
        rows, cols = np.transpose(pos)
        result[rows, cols] = RobotCommons.Tile.SOLUTION
    if initial:
        result[initial.x, initial.y] = RobotCommons.Tile.START
    if goal:
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

    cb = f.colorbar(a1, ax=ax1)
    cb.set_ticks(np.arange(len(RobotCommons.color_map)) * 36)
    cb.set_ticklabels(['GOAL', 'START', 'EMPTY', 'WALL', 'VISITED', 'SOLUTION', 'CURRENT', 'NEXT'])
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
        x, y, last = pluck(state, 'x', 'y', 'last_movement')
        t = self.visited_tiles
        max_x, max_y = np.shape(t)
        left, right, up, down = x - 1, x + 1, y - 1, y + 1

        def __can_go_up_left():
            return up > 0 and left > 0 and t[left, up] != RobotCommons.Tile.WALL and (
                    t[x, up] != RobotCommons.Tile.WALL or t[
                left, y] != RobotCommons.Tile.WALL) and last not in RobotCommons.up_left

        def __can_go_down_left():
            return down < max_y and left > 0 and t[left, down] != RobotCommons.Tile.WALL and (
                    t[left, y] != RobotCommons.Tile.WALL or t[x, down] != RobotCommons.Tile.WALL) and last not in RobotCommons.down_left

        def __can_go_up_right():
            return up > 0 and right < max_x and t[right, up] != RobotCommons.Tile.WALL and (
                    t[x, up] != RobotCommons.Tile.WALL or t[
                right, y] != RobotCommons.Tile.WALL) and last not in RobotCommons.up_right

        def __can_go_down_right():
            return down < max_y and right < max_x and t[right, down] != RobotCommons.Tile.WALL and (
                    t[right, y] != RobotCommons.Tile.WALL or t[x, down] != RobotCommons.Tile.WALL) and last not in RobotCommons.down_right

        def __can_go_up():
            return up > 0 and t[x, up] != RobotCommons.Tile.WALL and last not in RobotCommons.up

        def __can_go_down():
            return down < max_y and t[x, down] != RobotCommons.Tile.WALL and last not in RobotCommons.down

        def __can_go_left():
            return left > 0 and t[left, y] != RobotCommons.Tile.WALL and last not in RobotCommons.left

        def __can_go_right():
            return right < max_x and t[right, y] != RobotCommons.Tile.WALL and last not in RobotCommons.right

        self.step += 1
        if self.step > LAST_FRAME:
            plot_tile_map(fix_tile_map_after_solution(self.visited_tiles, self.initial, self.goal, None, state, None), False)
            plt.savefig('./img/{}/{}__{:09d}.png'.format(self.alg, self.alg, self.step))
            plt.cla()

        if self.alg != RobotCommons.BlindSearchMethods.DFS:  # DFS is implemented using a explicit stack, all the other variations of DFS are implemented using recursion
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

        self.__mark_visited(state)

    def __mark_visited(self, state):
        self.visited_tiles[state.x, state.y] = RobotCommons.Tile.VISITED
        return state.x, state.y

    def __move_up(self, s):
        visited_tile = self.__mark_visited(s)
        st = set()
        st.add(visited_tile)
        return State(s.x, s.y - 1, RobotCommons.Actions.MV_UP, s.already_visited | st)

    def __move_down(self, s):
        visited_tile = self.__mark_visited(s)
        st = set()
        st.add(visited_tile)
        return State(s.x, s.y + 1, RobotCommons.Actions.MV_DOWN, s.already_visited | st)

    def __move_left(self, s):
        visited_tile = self.__mark_visited(s)
        st = set()
        st.add(visited_tile)
        return State(s.x - 1, s.y, RobotCommons.Actions.MV_LEFT, s.already_visited | st)

    def __move_right(self, s):
        visited_tile = self.__mark_visited(s)
        st = set()
        st.add(visited_tile)
        return State(s.x + 1, s.y, RobotCommons.Actions.MV_RIGHT, s.already_visited | st)

    def __move_up_left(self, s):
        visited_tile = self.__mark_visited(s)
        st = set()
        st.add(visited_tile)
        return State(s.x - 1, s.y - 1, RobotCommons.Actions.MV_UP_LEFT, s.already_visited | st)

    def __move_down_left(self, s):
        visited_tile = self.__mark_visited(s)
        st = set()
        st.add(visited_tile)
        return State(s.x - 1, s.y + 1, RobotCommons.Actions.MV_DOWN_LEFT, s.already_visited | st)

    def __move_up_right(self, s):
        visited_tile = self.__mark_visited(s)
        st = set()
        st.add(visited_tile)
        return State(s.x + 1, s.y - 1, RobotCommons.Actions.MV_UP_RIGHT, s.already_visited | st)

    def __move_down_right(self, s):
        visited_tile = self.__mark_visited(s)
        st = set()
        st.add(visited_tile)
        return State(s.x + 1, s.y + 1, RobotCommons.Actions.MV_DOWN_RIGHT, s.already_visited | st)

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

        next_state = func(*args)
        if self.step > LAST_FRAME:
            plot_tile_map(fix_tile_map_after_solution(self.visited_tiles, self.initial, self.goal, None, state, next_state), False)
            plt.savefig('./img/{}/{}__{:09d}.png'.format(self.alg, self.alg, self.step))
            plt.cla()
        return next_state

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


class RobotBFS(Problem):

    def __init__(self, tiles, alg, initial, goal=None):
        super(RobotBFS, self).__init__(initial, goal)
        self.alg = alg
        self.visited_tiles = tiles.copy()
        self.step = 0

    def actions(self, state):
        x, y, last = pluck(state, 'x', 'y', 'last_movement')
        t = self.visited_tiles
        max_x, max_y = np.shape(t)
        left, right, up, down = x - 1, x + 1, y - 1, y + 1

        def __can_go_up_left():
            return up > 0 and left > 0 and 0 < x < max_x and 0 < y < max_y and t[left, up] != RobotCommons.Tile.WALL and (
                    t[x, up] != RobotCommons.Tile.WALL or t[left, y] != RobotCommons.Tile.WALL) and last not in RobotCommons.up_left

        def __can_go_down_left():
            return down < max_y and left > 0 and 0 < x < max_x and 0 < y < max_y and t[left, down] != RobotCommons.Tile.WALL and (
                    t[left, y] != RobotCommons.Tile.WALL or t[x, down] != RobotCommons.Tile.WALL) and last not in RobotCommons.down_left

        def __can_go_up_right():
            return up > 0 and right < max_x and 0 < x < max_x and 0 < y < max_y and t[right, up] != RobotCommons.Tile.WALL and (
                    t[x, up] != RobotCommons.Tile.WALL or t[
                right, y] != RobotCommons.Tile.WALL) and last not in RobotCommons.up_right

        def __can_go_down_right():
            return down < max_y and right < max_x and 0 < x < max_x and 0 < y < max_y and t[right, down] != RobotCommons.Tile.WALL and (
                    t[right, y] != RobotCommons.Tile.WALL or t[x, down] != RobotCommons.Tile.WALL) and last not in RobotCommons.down_right

        def __can_go_up():
            return up > 0 and t[x, up] != RobotCommons.Tile.WALL and last not in RobotCommons.up

        def __can_go_down():
            return down < max_y and t[x, down] != RobotCommons.Tile.WALL and last not in RobotCommons.down

        def __can_go_left():
            return left > 0 and t[left, y] != RobotCommons.Tile.WALL and last not in RobotCommons.left

        def __can_go_right():
            return right < max_x and t[right, y] != RobotCommons.Tile.WALL and last not in RobotCommons.right

        self.step += 1
        if self.step > LAST_FRAME:
            # plot_tile_map(fix_tile_map_after_solution(self.visited_tiles, self.initial, self.goal, None, state, None), False)
            plot_tile_map(fix_tile_map_after_solution(self.visited_tiles, self.initial, self.goal, None, state, None), False)
            plt.savefig('./img/{}/{}__{:09d}.png'.format(self.alg, self.alg, self.step))
            plt.cla()

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

        self.__mark_visited(state)

    def __mark_visited(self, state):
        self.visited_tiles[state.x, state.y] = RobotCommons.Tile.VISITED
        return state.x, state.y

    def __move_up(self, s):
        return State(s.x, s.y - 1, RobotCommons.Actions.MV_UP)

    def __move_down(self, s):
        return State(s.x, s.y + 1, RobotCommons.Actions.MV_DOWN)

    def __move_left(self, s):
        return State(s.x - 1, s.y, RobotCommons.Actions.MV_LEFT)

    def __move_right(self, s):
        return State(s.x + 1, s.y, RobotCommons.Actions.MV_RIGHT)

    def __move_up_left(self, s):
        return State(s.x - 1, s.y - 1, RobotCommons.Actions.MV_UP_LEFT)

    def __move_down_left(self, s):
        return State(s.x - 1, s.y + 1, RobotCommons.Actions.MV_DOWN_LEFT)

    def __move_up_right(self, s):
        return State(s.x + 1, s.y - 1, RobotCommons.Actions.MV_UP_RIGHT)

    def __move_down_right(self, s):
        return State(s.x + 1, s.y + 1, RobotCommons.Actions.MV_DOWN_RIGHT)

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

        next_state = func(*args)
        if self.step > LAST_FRAME:
            plot_tile_map(fix_tile_map_after_solution(self.visited_tiles, self.initial, self.goal, None, state, next_state), False)
            plt.savefig('./img/{}/{}__{:09d}.png'.format(self.alg, self.alg, self.step))
            plt.cla()
        return next_state

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
        sx, sy = pluck(start, 'x', 'y')
        gx, gy = pluck(goal, 'x', 'y')
        self.tiles = np.array(generate_maze(width, height, walls))
        self.tiles[sx, sy] = RobotCommons.Tile.START
        self.tiles[gx, gy] = RobotCommons.Tile.GOAL
        self.instance = problem(self.tiles, alg, StateBFS(sx, sy), goal)


if __name__ == '__main__':
    # width, height = 10, 10
    # start = StateBFS(1, 1)
    # goal = StateBFS(8, 8)
    # walls = [(i, 2) for i in range(0, 8)] + [(i, 7) for i in range(9, 2, -1)]

    width, height = 60, 60
    start = State(10, 10)
    goal = State(50, 50)
    walls = [(i, 20) for i in range(0, 40)] + [(i, 40) for i in range(59, 20, -1)]

    # width, height = 10, 10
    # start = State(1, 1, 90, None, None)
    # goal = State(19, 19, 0, None, None)
    # walls = 1

    for alg in RobotCommons.BlindSearchMethods:
        problem = RobotProblemFactory(RobotBFS, alg, width, height, start, goal, walls)

        plot_tile_map(fix_tile_map_after_solution(problem.tiles, problem.start, problem.goal, None, None, None), False)
        plt.savefig('./img/{}/{}__{:09d}.png'.format(alg, alg, 0))
        plt.cla()
        e, solution = elapsed(lambda: alg.method(problem.instance))
        print('Robot ', alg, ' found solution in ', e, 's : ', solution.path())
        plot_tile_map(fix_tile_map_after_solution(problem.instance.visited_tiles, problem.start, problem.goal, solution), False)
        plt.savefig('./img/{}/{}__{:09d}.png'.format(alg, alg, problem.instance.step + 1))
        plt.cla()
        print(alg)

    problem = RobotProblemFactory(Robot, "A_star_manhattan", width, height, start, goal, walls)
    solution = astar_search(problem.instance, problem.instance.manhattan_distance)
    print('A* using manhattan distance: ', solution.path())
    plot_tile_map(fix_tile_map_after_solution(problem.instance.visited_tiles, problem.start, problem.goal, solution))

    problem = RobotProblemFactory(Robot, "A_star_euclidian_sqr", width, height, start, goal, walls)
    solution = astar_search(problem.instance, problem.instance.euclidean_distance)
    print('A* using sqr euclidian distance: ', solution.path())
    plot_tile_map(fix_tile_map_after_solution(problem.instance.visited_tiles, problem.start, problem.goal, solution))
