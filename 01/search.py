#!/usr/bin/env python
# coding: utf-8

import time
# Needed to hide warnings in the matplotlib sections
import warnings
from enum import Enum, IntEnum, auto

import matplotlib.pyplot as plt
import numpy as np

from maze_generator import *
from search import Problem, Node, InstrumentedProblem, breadth_first_graph_search, depth_first_graph_search, astar_search

warnings.filterwarnings("ignore")


# ======= Utilities that make my life easier ======

def pluck(dict, *args):
    """
    dict unpacking
    :param dict: the dict to unpack
    :param args: keys to be unpacked
    :return: a tuple of values in `args` order
    """
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
    """
    Robot state representation, just like a tuple, but disconsiders `last_movement` when comparing with another `State`
    """
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
        """
        Defines the current state of a map position
        Used for plotting
        """
        GOAL = 0
        START = 1
        EMPTY = 2
        WALL = 3
        VISITED = 4
        SOLUTION = 5
        CURRENT = 6
        NEXT = 7

    class Actions(Enum):
        """
        ""Set"" of all possible robot actions
        """
        MV_UP = auto()
        MV_RIGHT = auto()
        MV_LEFT = auto()
        MV_DOWN = auto()
        MV_UP_LEFT = auto()
        MV_UP_RIGHT = auto()
        MV_DOWN_LEFT = auto()
        MV_DOWN_RIGHT = auto()

    class Directions(Enum):
        """
        ""Set"" of all possible directions the robot can go
        """
        UP = auto()
        RIGHT = auto()
        LEFT = auto()
        DOWN = auto()
        UP_LEFT = auto()
        UP_RIGHT = auto()
        DOWN_LEFT = auto()
        DOWN_RIGHT = auto()

    """
    Color map for plotting
    """
    color_map = {
        Tile.EMPTY: np.array([[255, 255, 255, 0]]) / 255.0,
        Tile.WALL: np.array([[65, 63, 62, 255]]) / 255.0,
        Tile.SOLUTION: np.array([[24, 144, 136, 255]]) / 255.0,
        Tile.START: np.array([[117, 151, 143, 255]]) / 255.0,
        Tile.VISITED: np.array([[209, 187, 161, 150]]) / 255.0,
        Tile.GOAL: np.array([[235, 101, 89, 255]]) / 255.0,
        Tile.CURRENT: np.array([[184, 15, 10, 175]]) / 255.0,
        Tile.NEXT: np.array([114, 133, 165, 175]) / 255.0
    }

    """
    Immutable sets to validate `last_position`
    """
    up = frozenset((Actions.MV_DOWN, Actions.MV_DOWN_RIGHT, Actions.MV_DOWN_LEFT))
    down = frozenset((Actions.MV_UP, Actions.MV_UP_RIGHT, Actions.MV_UP_LEFT))
    right = frozenset((Actions.MV_LEFT, Actions.MV_UP_LEFT, Actions.MV_DOWN_LEFT))
    left = frozenset((Actions.MV_RIGHT, Actions.MV_UP_RIGHT, Actions.MV_DOWN_RIGHT))
    up_left = frozenset((Actions.MV_DOWN, Actions.MV_RIGHT, Actions.MV_DOWN_RIGHT))
    up_right = frozenset((Actions.MV_DOWN, Actions.MV_LEFT, Actions.MV_DOWN_LEFT))
    down_left = frozenset((Actions.MV_UP, Actions.MV_RIGHT, Actions.MV_UP_RIGHT))
    down_right = frozenset((Actions.MV_UP, Actions.MV_LEFT, Actions.MV_UP_LEFT))


def fix_tile_map_after_solution(tiles, initial, goal, solution, current=None, next=None):
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
    width, height = np.shape(tiles)
    visited_y, visited_x = np.where(tiles == RobotCommons.Tile.VISITED)
    solution_y, solution_x = np.where(tiles == RobotCommons.Tile.SOLUTION)
    goal_y, goal_x = np.where(tiles == RobotCommons.Tile.GOAL)
    start_y, start_x = np.where(tiles == RobotCommons.Tile.START)
    walls_y, walls_x = np.where(tiles == RobotCommons.Tile.WALL)
    current_y, current_x = np.where(tiles == RobotCommons.Tile.CURRENT)
    next_y, next_x = np.where(tiles == RobotCommons.Tile.NEXT)

    border_walls_y = np.concatenate((np.repeat(-1, width), np.repeat(-1, height), np.repeat(width, width), np.repeat(height, height), np.arange(width),
                                     np.arange(height), np.arange(width), np.arange(height)))
    border_walls_x = np.concatenate((np.arange(width), np.arange(height), np.arange(width), np.arange(height), np.repeat(-1, width), np.repeat(-1, height),
                                     np.repeat(width, width), np.repeat(height, height)))

    f, ax1 = plt.subplots(nrows=1)
    ax1.scatter(x=visited_x, y=visited_y, c=RobotCommons.color_map.get(RobotCommons.Tile.VISITED), marker='x')
    ax1.scatter(x=solution_x, y=solution_y, c=RobotCommons.color_map.get(RobotCommons.Tile.SOLUTION), marker='.')
    ax1.scatter(x=np.concatenate((walls_x, border_walls_x)), y=np.concatenate((walls_y, border_walls_y)),
                c=RobotCommons.color_map.get(RobotCommons.Tile.WALL), marker='s')
    ax1.scatter(x=goal_x, y=goal_y, c=RobotCommons.color_map.get(RobotCommons.Tile.GOAL), marker='*')
    ax1.scatter(x=start_x, y=start_y, c=RobotCommons.color_map.get(RobotCommons.Tile.START), marker='X')
    ax1.scatter(x=current_x, y=current_y, c=RobotCommons.color_map.get(RobotCommons.Tile.CURRENT), marker='o')
    ax1.scatter(x=next_x, y=next_y, c=RobotCommons.color_map.get(RobotCommons.Tile.NEXT), marker='x')
    # Major ticks
    width_step = 10 ** np.log10(width - 1).astype(int)
    height_step = 10 ** np.log10(height - 1).astype(int)

    ax1.set_xticks(np.arange(width + width_step, step=width_step))
    ax1.set_yticks(np.arange(height + width_step, step=height_step))

    # Labels for major ticks
    ax1.set_xticklabels(np.arange(width + width_step, step=width_step))
    ax1.set_yticklabels(np.arange(height + width_step, step=height_step))

    # Minor ticks
    ax1.set_xticks(np.arange(width + width_step, step=width_step) - (width_step * .5), minor=True)
    ax1.set_yticks(np.arange(height + width_step, step=height_step) - (height_step * .5), minor=True)

    # Gridlines based on minor ticks
    line_color = (90 / 255., 90 / 255., 90 / 255.)
    ax1.grid(which='minor', color=line_color, linestyle='--', linewidth=1)

    plt.setp((ax1,), xticks=np.arange(width + width_step, step=width_step), yticks=np.arange(height + height_step, step=height_step))
    if show_img:
        plt.show()


class Robot(Problem):

    def __init__(self, tiles, walls, alg, initial, goal=None):
        super(Robot, self).__init__(initial, goal)
        self.tiles = tiles  # we keep the map here just to mark all visited positions, for plotting
        self.alg = alg  # we need the search method to decide if we're going to shuffle the action list or not :P
        self.height, self.width = np.shape(self.tiles)
        self.walls = walls

    def actions(self, state):
        x, y, last = pluck(state, 'x', 'y', 'last_movement')
        max_x, max_y = self.width, self.height
        left, right, up, down = x - 1, x + 1, y - 1, y + 1
        positions = {
            RobotCommons.Directions.UP: (up, x),
            RobotCommons.Directions.RIGHT: (y, right),
            RobotCommons.Directions.LEFT: (y, left),
            RobotCommons.Directions.DOWN: (down, x),
            RobotCommons.Directions.UP_LEFT: (up, left),
            RobotCommons.Directions.UP_RIGHT: (up, right),
            RobotCommons.Directions.DOWN_LEFT: (down, left),
            RobotCommons.Directions.DOWN_RIGHT: (down, right)
        }

        """
        the `__can_go_<direction>()` methods validates map bounds and walls constraints
        """

        def __can_go_up_left():
            return up > 0 and left > 0 and 0 <= x < max_x and 0 <= y < max_y and positions.get(RobotCommons.Directions.UP_LEFT) not in self.walls and (
                    positions.get(RobotCommons.Directions.UP) not in self.walls or positions.get(
                RobotCommons.Directions.LEFT) not in self.walls) and last not in RobotCommons.up_left

        def __can_go_down_left():
            return down < max_y and left > 0 and 0 <= x < max_x and 0 <= y < max_y and positions.get(
                RobotCommons.Directions.DOWN_LEFT) not in self.walls and (
                           positions.get(RobotCommons.Directions.DOWN) not in self.walls or positions.get(
                       RobotCommons.Directions.LEFT) not in self.walls) and last not in RobotCommons.down_left

        def __can_go_up_right():
            return up > 0 and right < max_x and 0 <= x < max_x and 0 <= y < max_y and positions.get(
                RobotCommons.Directions.UP_RIGHT) not in self.walls and (
                           positions.get(RobotCommons.Directions.UP) not in self.walls or positions.get(
                       RobotCommons.Directions.RIGHT) not in self.walls) and last not in RobotCommons.up_right

        def __can_go_down_right():
            return down < max_y and right < max_x and 0 <= x < max_x and 0 <= y < max_y and positions.get(
                RobotCommons.Directions.DOWN_RIGHT) not in self.walls and (
                           positions.get(RobotCommons.Directions.DOWN) not in self.walls or positions.get(
                       RobotCommons.Directions.RIGHT) not in self.walls) and last not in RobotCommons.down_right

        def __can_go_up():
            return up > 0 and positions.get(RobotCommons.Directions.UP) not in self.walls and last not in RobotCommons.up

        def __can_go_down():
            return down < max_y and positions.get(RobotCommons.Directions.DOWN) not in self.walls and last not in RobotCommons.down

        def __can_go_left():
            return left > 0 and positions.get(RobotCommons.Directions.LEFT) not in self.walls and last not in RobotCommons.left

        def __can_go_right():
            return right < max_x and positions.get(RobotCommons.Directions.RIGHT) not in self.walls and last not in RobotCommons.right

        actions_list = []

        if __can_go_down_left():
            actions_list.append(RobotCommons.Actions.MV_DOWN_LEFT)

        if __can_go_down_right():
            actions_list.append(RobotCommons.Actions.MV_DOWN_RIGHT)

        if __can_go_up_left():
            actions_list.append(RobotCommons.Actions.MV_UP_LEFT)

        if __can_go_up_right():
            actions_list.append(RobotCommons.Actions.MV_UP_RIGHT)

        if __can_go_up():
            actions_list.append(RobotCommons.Actions.MV_UP)

        if __can_go_down():
            actions_list.append(RobotCommons.Actions.MV_DOWN)

        if __can_go_right():
            actions_list.append(RobotCommons.Actions.MV_RIGHT)

        if __can_go_left():
            actions_list.append(RobotCommons.Actions.MV_LEFT)

        if alg == "DFS":
            random.shuffle(actions_list)

        self.tiles[state.x, state.y] = RobotCommons.Tile.VISITED

        return actions_list

    @staticmethod
    def __move_up(s):
        return State(s.x, s.y - 1, RobotCommons.Actions.MV_UP)

    @staticmethod
    def __move_down(s):
        return State(s.x, s.y + 1, RobotCommons.Actions.MV_DOWN)

    @staticmethod
    def __move_left(s):
        return State(s.x - 1, s.y, RobotCommons.Actions.MV_LEFT)

    @staticmethod
    def __move_right(s):
        return State(s.x + 1, s.y, RobotCommons.Actions.MV_RIGHT)

    @staticmethod
    def __move_up_left(s):
        return State(s.x - 1, s.y - 1, RobotCommons.Actions.MV_UP_LEFT)

    @staticmethod
    def __move_down_left(s):
        return State(s.x - 1, s.y + 1, RobotCommons.Actions.MV_DOWN_LEFT)

    @staticmethod
    def __move_up_right(s):
        return State(s.x + 1, s.y - 1, RobotCommons.Actions.MV_UP_RIGHT)

    @staticmethod
    def __move_down_right(s):
        return State(s.x + 1, s.y + 1, RobotCommons.Actions.MV_DOWN_RIGHT)

    def result(self, state, action):
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
        return np.math.sqrt((state.x - self.goal.x) ** 2 + (state.y - self.goal.y) ** 2)


def generate_maze(width, height, walls):
    tiles = np.ones((width, height)).astype(int) * RobotCommons.Tile.EMPTY
    if walls:
        rows, cols = np.transpose(walls)
        tiles[rows, cols] = RobotCommons.Tile.WALL
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
        wy, wx = np.where(self.tiles == RobotCommons.Tile.WALL)
        self.instance = InstrumentedProblem(problem(self.tiles, set(zip(wx, wy)), alg, State(sx, sy), goal))


if __name__ == '__main__':
    width, height = 60, 60
    start = State(10, 10)
    goal = State(50, 50)
    walls = [(i, 20) for i in range(0, 40)] + [(i, 40) for i in range(59, 20, -1)]

    alg = "optimal"
    problem = RobotProblemFactory(Robot, alg, width, height, start, goal, walls)
    solution = problem
    solution.path = lambda: np.unique(
        np.array([Node(State(10, 10)), Node(State(11, 10)), Node(State(12, 11)), Node(State(13, 11)), Node(State(14, 11)), Node(State(15, 12)),
                  Node(State(16, 12)), Node(State(17, 12)),
                  Node(State(18, 12)),
                  Node(State(19, 13)), Node(State(20, 13)), Node(State(21, 13)), Node(State(22, 14)), Node(State(23, 14)), Node(State(24, 14)),
                  Node(State(25, 15)), Node(State(26, 15)),
                  Node(State(27, 15)),
                  Node(State(28, 16)), Node(State(29, 16)), Node(State(30, 16)), Node(State(31, 17)), Node(State(32, 17)), Node(State(33, 17)),
                  Node(State(34, 18)),
                  Node(State(35, 18)),
                  Node(State(36, 18)), Node(State(37, 19)), Node(State(38, 19)), Node(State(39, 19)), Node(State(40, 20)), Node(State(39, 21)),
                  Node(State(38, 22)), Node(State(37, 23)),
                  Node(State(36, 24)), Node(State(35, 25)), Node(State(34, 26)), Node(State(33, 27)), Node(State(32, 28)), Node(State(31, 29)),
                  Node(State(30, 30)), Node(State(29, 31)),
                  Node(State(28, 32)), Node(State(27, 33)), Node(State(26, 34)), Node(State(25, 35)), Node(State(24, 36)), Node(State(23, 37)),
                  Node(State(22, 38)), Node(State(21, 39)),
                  Node(State(20, 40)), Node(State(21, 41)), Node(State(22, 41)), Node(State(23, 41)), Node(State(24, 42)), Node(State(25, 42)),
                  Node(State(26, 42)), Node(State(27, 42)),
                  Node(State(28, 43)), Node(State(29, 43)), Node(State(30, 43)), Node(State(31, 44)), Node(State(32, 44)), Node(State(33, 44)),
                  Node(State(34, 45)), Node(State(35, 45)),
                  Node(State(36, 45)), Node(State(37, 46)), Node(State(38, 46)), Node(State(39, 46)), Node(State(40, 47)), Node(State(41, 47)),
                  Node(State(42, 47)), Node(State(43, 47)),
                  Node(State(44, 48)), Node(State(45, 48)), Node(State(46, 48)), Node(State(47, 49)), Node(State(48, 49)), Node(State(49, 50)),
                  Node(State(50, 50))]))

    plot_tile_map(fix_tile_map_after_solution(problem.tiles, problem.start, problem.goal, solution, None, None), False)
    plt.savefig('./img/{0}/{0}__solution.png'.format(alg))
    plt.cla()
    e = 0
    print('Robot ', alg, ' found solution in ', e, 's : ', solution.path())
    print('path length: ', len(solution.path()))

    alg = "BFS"
    problem = RobotProblemFactory(Robot, alg, width, height, start, goal, walls)

    plot_tile_map(fix_tile_map_after_solution(problem.tiles, problem.start, problem.goal, None, None, None), False)
    plt.savefig('./img/{0}.png'.format("map"))
    plt.cla()

    e, solution = elapsed(lambda: breadth_first_graph_search(problem.instance))
    print('Robot ', alg, ' found solution in ', e, 's : ', solution.path())
    print('path length: ', len(solution.path()))
    print('path cost: ', solution.path_cost)
    print('goal checks: ', problem.instance.goal_tests)
    print('states explored: ', problem.instance.succs)
    print('actions executed: ', problem.instance.states)
    plot_tile_map(fix_tile_map_after_solution(problem.tiles, problem.start, problem.goal, solution), False)
    plt.savefig('./img/{0}/{0}__solution.png'.format(alg))
    plt.cla()

    # ======= Utilities that make my life easier ======

    alg = "DFS"
    problem = RobotProblemFactory(Robot, alg, width, height, start, goal, walls)

    e, solution = elapsed(lambda: depth_first_graph_search(problem.instance))
    print('Robot ', alg, ' found solution in ', e, 's : ', solution.path())
    print('path length: ', len(solution.path()))
    print('path cost: ', solution.path_cost)
    print('goal checks: ', problem.instance.goal_tests)
    print('states explored: ', problem.instance.succs)
    print('actions executed: ', problem.instance.states)
    plot_tile_map(fix_tile_map_after_solution(problem.tiles, problem.start, problem.goal, solution), False)
    plt.savefig('./img/{0}/{0}__solution.png'.format(alg))
    plt.cla()

    # ======= Utilities that make my life easier ======

    alg = "A_star_manhattan"
    problem = RobotProblemFactory(Robot, alg, width, height, start, goal, walls)
    e, solution = elapsed(lambda: astar_search(problem.instance, problem.instance.problem.manhattan_distance))
    print('Robot A* (manhattan dist) found solution in ', e, 's : ', solution.path())
    print('path length: ', len(solution.path()))
    print('path cost: ', solution.path_cost)
    print('goal checks: ', problem.instance.goal_tests)
    print('states explored: ', problem.instance.succs)
    print('actions executed: ', problem.instance.states)
    plot_tile_map(fix_tile_map_after_solution(problem.tiles, problem.start, problem.goal, solution), False)
    plt.savefig('./img/{0}/{0}__solution.png'.format(alg))
    plt.cla()

    # ======= Utilities that make my life easier ======

    alg = "A_star_euclidean"
    problem = RobotProblemFactory(Robot, alg, width, height, start, goal, walls)
    e, solution = elapsed(lambda: astar_search(problem.instance, problem.instance.problem.euclidean_distance))
    print('Robot A* (euclidean dist) found solution in ', e, 's : ', solution.path())
    print('path length: ', len(solution.path()))
    print('path cost: ', solution.path_cost)
    print('goal checks: ', problem.instance.goal_tests)
    print('states explored: ', problem.instance.succs)
    print('actions executed: ', problem.instance.states)
    plot_tile_map(fix_tile_map_after_solution(problem.tiles, problem.start, problem.goal, solution), False)
    plt.savefig('./img/{0}/{0}__solution.png'.format(alg))
    plt.cla()
