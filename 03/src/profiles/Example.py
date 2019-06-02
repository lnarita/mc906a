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
                        (self.kawaii[self.higher]) |
                        (self.beautiful[self.higher] | self.beautiful[self.highest])
                ) &
                (
                        (
                                (self.smart[self.high] | self.smart[self.higher] | self.smart[self.highest]) &
                                (self.baka[self.lowest] | self.baka[self.low] | self.baka[self.high])
                        ) |
                        (self.innocent[self.lower] | self.innocent[self.higher] | self.innocent[self.highest]) |
                        (self.lewd[self.high] | self.lewd[self.higher]) |
                        (
                                (self.female[self.high] | self.female[self.higher] | self.female[self.highest]) &
                                (
                                        (self.tsundere[self.lowest]) &
                                        (self.catgirl[self.lowest]) &
                                        (self.clumsy[self.lowest]) &
                                        (
                                                (self.busty[self.high] | self.busty[self.higher]) |
                                                (self.flat[self.high] | self.flat[self.higher])
                                        ) &
                                        (self.yandere[self.low] | self.yandere[self.high]) |
                                        (self.cheerful[self.high] & self.gloomy[self.higher])
                                )
                        ) |
                        (
                                (self.male[self.high] | self.male[self.higher] | self.male[self.highest]) &
                                (self.flat[self.lowest]) &
                                (
                                        (
                                                (self.m[self.high] | self.m[self.higher]) |
                                                (self.s[self.high] | self.s[self.higher])
                                        ) &
                                        (self.yandere[self.high] | self.yandere[self.higher] | self.yandere[self.highest])
                                )
                        )
                )
        ), self.likeness['high']
                         )

    def is_waifu(self):
        return ctrl.Rule(super().is_waifu() & (
                (
                        (self.kawaii[self.high] | self.kawaii[self.higher] | self.kawaii[self.highest]) |
                        (self.beautiful[self.high] | self.beautiful[self.higher] | self.beautiful[self.highest])
                ) &
                (
                        (
                                (self.smart[self.high] | self.smart[self.higher] | self.smart[self.highest]) &
                                (self.baka[self.lowest] | self.baka[self.lower] | self.baka[self.low])
                        ) |
                        (self.innocent[self.low] | self.innocent[self.high] | self.innocent[self.higher]) |
                        (self.lewd[self.high] | self.lewd[self.higher]) |
                        (
                                (self.female[self.high] | self.female[self.higher] | self.female[self.highest]) &
                                (
                                        (self.tsundere[self.lowest]) |
                                        (self.catgirl[self.lowest]) |
                                        (self.clumsy[self.lowest]) |
                                        (self.plain[self.high]) |
                                        (self.busty[self.high] | self.busty[self.higher]) |
                                        (self.flat[self.high] | self.flat[self.higher]) |
                                        (self.yandere[self.low] | self.yandere[self.high])
                                )
                        ) |
                        (
                                (self.male[self.high] | self.male[self.higher] | self.male[self.highest]) &
                                (self.flat[self.lowest]) &
                                (
                                        (self.m[self.high]) |
                                        (self.plain[self.low]) |
                                        (self.megane[self.high]) |
                                        (self.s[self.high] | self.s[self.higher]) |
                                        (self.yandere[self.high] | self.yandere[self.higher] | self.yandere[self.highest])
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
                        (self.smart[self.lowest] | self.smart[self.lower] | self.smart[self.low]) &
                        (self.baka[self.higher] | self.baka[self.highest])
                ) |
                (self.short[self.low] | self.short[self.high] | self.short[self.higher] | self.short[self.highest]) |
                (self.tall[self.high] | self.tall[self.higher] | self.tall[self.highest]) |
                (self.innocent[self.higher] | self.innocent[self.highest]) |
                (self.lewd[self.lowest] | self.lewd[self.highest]) |
                (
                        (self.female[self.high] | self.female[self.higher] | self.female[self.highest]) &
                        (
                                (self.tsundere[self.high] | self.tsundere[self.higher] | self.tsundere[self.highest]) |
                                (self.catgirl[self.high] | self.catgirl[self.higher] | self.catgirl[self.highest]) |
                                (self.clumsy[self.high] | self.clumsy[self.higher] | self.clumsy[self.highest]) |
                                (self.s[self.higher] | self.s[self.highest]) |
                                (self.m[self.higher] | self.m[self.highest]) |
                                (self.flashy[self.high] | self.flashy[self.higher] | self.flashy[self.highest]) |
                                (self.busty[self.highest]) |
                                (self.flat[self.lowest]) |
                                (self.yandere[self.higher] | self.yandere[self.highest]) |
                                (self.cheerful[self.higher] | self.cheerful[self.highest])
                        )
                ) |
                (
                        (self.male[self.high] | self.male[self.higher] | self.male[self.highest]) &
                        (
                                (self.m[self.highest]) |
                                (self.plain[self.higher] | self.plain[self.highest]) |
                                (self.busty[self.high] | self.busty[self.higher] | self.busty[self.highest])
                        )
                )
        ), self.likeness['low'])
