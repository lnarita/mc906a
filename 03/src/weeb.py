from functools import reduce
from itertools import product

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from src.commons import Map


class Weeb(Map):
    def __init__(self, waifu_traits):
        super().__init__()
        self.__control_system = None
        self._traits = waifu_traits
        self._labels = []

        def default_setup():
            pass

        def add_trait(trait):
            self[trait] = ctrl.Antecedent(np.arange(0, 11, 1), trait)
            setup_method_name = "setup_{}".format(trait)
            self[setup_method_name] = default_setup

        for trait in waifu_traits:
            add_trait(trait)
        self.likeness = ctrl.Consequent(np.arange(0, 11, 1), 'likeness')
        self.likeness.automf(3, variable_type='quant')

    def __ignore_all_traits(self):
        if not self.__ignore_all_rule:
            self.__ignore_all_rule = reduce(lambda a, b: self[b[0]][b[1]] if not a else a | self[b[0]][b[1]], product(self._traits, self._labels), False)
        return self.__ignore_all_rule

    def is_best_gril(self):
        return self.__ignore_all_traits()

    def is_waifu(self):
        return self.__ignore_all_traits()

    def is_trash(self):
        return self.__ignore_all_traits()

    def predict(self, gril):
        if not self.__control_system:
            for trait in self._traits:
                self["setup_{}".format(trait)]()
            self.__control_system = ctrl.ControlSystem([self.is_best_gril(), self.is_waifu(), self.is_trash()])

        control_simulation = ctrl.ControlSystemSimulation(self.__control_system)
        for trait in self._traits:
            attr = gril[trait]
            control_simulation.input[trait] = attr

        control_simulation.compute()
        return control_simulation

# self.kawaii = ctrl.Antecedent(np.arange(0, 10, 1), 'kawaii')
# self.beautiful = ctrl.Antecedent(np.arange(0, 10, 1), 'beautiful')
# self.slim = ctrl.Antecedent(np.arange(0, 10, 1), 'slim')
# self.chubby = ctrl.Antecedent(np.arange(0, 10, 1), 'chubby')
# self.smart = ctrl.Antecedent(np.arange(0, 10, 1), 'smart')
# self.baka = ctrl.Antecedent(np.arange(0, 10, 1), 'baka')
# self.short = ctrl.Antecedent(np.arange(0, 10, 1), 'short')
# self.tall = ctrl.Antecedent(np.arange(0, 10, 1), 'tall')
# self.rich = ctrl.Antecedent(np.arange(0, 10, 1), 'rich')
# self.poor = ctrl.Antecedent(np.arange(0, 10, 1), 'poor')
# self.innocent = ctrl.Antecedent(np.arange(0, 10, 1), 'innocent')
# self.lewd = ctrl.Antecedent(np.arange(0, 10, 1), 'lewd')
# self.female = ctrl.Antecedent(np.arange(0, 10, 1), 'female')
# self.male = ctrl.Antecedent(np.arange(0, 10, 1), 'male')
# self.s = ctrl.Antecedent(np.arange(0, 10, 1), 's')
# self.m = ctrl.Antecedent(np.arange(0, 10, 1), 'm')
# self.clumsy = ctrl.Antecedent(np.arange(0, 10, 1), 'clumsy')
# self.skilled = ctrl.Antecedent(np.arange(0, 10, 1), 'skilled')
# self.cheerful = ctrl.Antecedent(np.arange(0, 10, 1), 'cheerful')
# self.gloomy = ctrl.Antecedent(np.arange(0, 10, 1), 'gloomy')
# self.flashy = ctrl.Antecedent(np.arange(0, 10, 1), 'flashy')
# self.plain = ctrl.Antecedent(np.arange(0, 10, 1), 'plain')
# self.flat = ctrl.Antecedent(np.arange(0, 10, 1), 'flat')
# self.busty = ctrl.Antecedent(np.arange(0, 10, 1), 'busty')
# self.megane = ctrl.Antecedent(np.arange(0, 10, 1), 'megane')
# self.catgirl = ctrl.Antecedent(np.arange(0, 10, 1), 'catgirl')
# self.tomboy = ctrl.Antecedent(np.arange(0, 10, 1), 'tomboy')
# self.trap = ctrl.Antecedent(np.arange(0, 10, 1), 'trap')
# self.maid = ctrl.Antecedent(np.arange(0, 10, 1), 'maid')
# self.butler = ctrl.Antecedent(np.arange(0, 10, 1), 'butler')
# self.tsundere = ctrl.Antecedent(np.arange(0, 10, 1), 'tsundere')
# self.yandere = ctrl.Antecedent(np.arange(0, 10, 1), 'yandere')
