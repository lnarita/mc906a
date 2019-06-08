import skfuzzy as fuzz
from skfuzzy import control as ctrl

from src.config import TRAITS
from src.weeb import Weeb


class ShortStacksLover(Weeb):
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
                        (self.kawaii[self.higher] | self.kawaii[self.highest]) |
                        (self.beautiful[self.higher] | self.beautiful[self.highest])
                ) &
                (
                        (
                                (self.smart[self.lowest] | self.smart[self.lower]) &
                                (self.baka[self.high] | self.baka[self.higher] | self.baka[self.highest])
                        ) |
                        (self.short[self.high] | self.short[self.higher] | self.short[self.highest]) |
                        (self.innocent[self.high] | self.innocent[self.higher] | self.innocent[self.highest]) |
                        (self.lewd[self.lowest]) |
                        (
                                (self.female[self.high] | self.female[self.higher] | self.female[self.highest]) &
                                (
                                        (self.tsundere[self.highest]) &
                                        (self.catgirl[self.lowest]) &
                                        (self.clumsy[self.highest]) &
                                        (
                                                (self.busty[self.higher] | self.busty[self.highest]) |
                                                (self.flat[self.lowest])
                                        ) &
                                        (self.yandere[self.lowest]) |
                                        (self.cheerful[self.higher] & self.gloomy[self.lower])
                                )
                        )
                )
        ), self.likeness['high'])

    def is_waifu(self):
        return ctrl.Rule(super().is_waifu() & (
                (
                        (self.kawaii[self.high] | self.kawaii[self.higher] | self.kawaii[self.highest]) |
                        (self.beautiful[self.high] | self.beautiful[self.higher] | self.beautiful[self.highest])
                ) &
                (
                        (
                                (self.baka[self.high] | self.baka[self.higher] | self.baka[self.highest])
                        ) |
                        (self.innocent[self.high]) |
                        (self.lewd[self.lower] | self.lewd[self.low]) |
                        (
                                (self.female[self.high] | self.female[self.higher] | self.female[self.highest]) &
                                (
                                        (self.tsundere[self.higher]) |
                                        (self.clumsy[self.low]) |
                                        (self.plain[self.high]) |
                                        (self.busty[self.high] | self.busty[self.higher]) |
                                        (self.flat[self.low] | self.flat[self.lower]) |
                                        (self.cheerful[self.high] | self.cheerful[self.higher]) |
                                        (self.yandere[self.lowest])
                                )
                        )
                )
        ), self.likeness['average'])

    def is_trash(self):
        return ctrl.Rule(super().is_trash() & (
                (
                        (self.kawaii[self.lowest] | self.kawaii[self.lower] | self.kawaii[self.low]) &
                        (self.beautiful[self.lowest] | self.beautiful[self.lower] | self.beautiful[self.low])
                ) |
                (
                        (self.smart[self.highest] | self.smart[self.higher] | self.smart[self.high]) &
                        (self.baka[self.lowest])
                ) |
                (self.tall[self.high] | self.tall[self.higher] | self.tall[self.highest]) |
                (self.lewd[self.high]) |
                (self.male[self.high] | self.male[self.higher] | self.male[self.highest]) &
                (
                        (self.female[self.high] | self.female[self.higher] | self.female[self.highest]) &
                        (
                                (self.catgirl[self.high] | self.catgirl[self.higher] | self.catgirl[self.highest]) |
                                (self.s[self.high] | self.s[self.higher] | self.s[self.highest]) |
                                (self.flashy[self.high] | self.flashy[self.higher] | self.flashy[self.highest]) |
                                (self.yandere[self.high] | self.yandere[self.higher] | self.yandere[self.highest]) |
                                (self.gloomy[self.higher] | self.gloomy[self.highest])
                        )
                )
        ), self.likeness['low'])
