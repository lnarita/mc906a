import skfuzzy as fuzz
from skfuzzy import control as ctrl

from src.config import TRAITS
from src.weeb import Weeb


class ExampleProfile(Weeb):
    def __init__(self):
        super().__init__(TRAITS)

        def default_setup_fun(trait):
            return lambda: self[trait].automf(6, names=self._labels)

        self.lowest = 'E'
        self.lower = 'D'
        self.low = 'C'
        self.high = 'B'
        self.higher = 'A'
        self.highest = 'EX'

        self._labels = [self.lowest, self.lower, self.low, self.high, self.higher, self.highest]

        for trait in self._traits:
            self["setup_{}".format(trait)] = default_setup_fun(trait)

    def is_best_gril(self):
        return ctrl.Rule(super().is_best_gril() & (
                (
                        self.kawaii[self.highest] |
                        (self.innocent[self.high] | self.lewd[self.lower] | self.lewd[self.low]) |
                        ((self.male[self.higher] & self.male[self.highest]) & (
                            (self.s[self.high] | self.s[self.low]))) |
                        (self.trap[self.highest] | self.tomboy[self.higher])
                ) &
                (
                        self.tsundere[self.high] | self.tsundere[self.higher] | self.tsundere[self.highest])
        ), self.likeness['high']
                         )

    def is_waifu(self):
        return ctrl.Rule(super().is_waifu() & (
                (self.kawaii[self.higher] | self.kawaii[self.highest]) &
                (self.megane[self.higher]) |
                ((self.male[self.higher] & self.male[self.highest]) &
                 ~(self.yandere[self.high] | self.yandere[self.higher] | self.yandere[self.highest])) |
                (self.maid[self.higher])
        ), self.likeness['average'])

    def is_trash(self):
        return ctrl.Rule(super().is_trash() & (
                (self.catgirl[self.higher] | self.catgirl[self.highest]) |
                self.kawaii[self.lowest] |
                self.female[self.highest] |
                (self.flashy[self.higher] | self.flashy[self.highest]) |
                ~(self.tsundere[self.high] | self.tsundere[self.higher] | self.tsundere[self.highest])
        ), self.likeness['low'])
