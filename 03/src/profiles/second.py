import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from src.config import TRAITS
from src.weeb import Weeb


class LewdAmazonLover(Weeb):
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
                    (self.beautiful[self.high] | self.beautiful[self.higher] | self.beautiful[self.highest])
                ) &
                (
                        (
                                (self.smart[self.highest] | self.smart[self.higher]) &
                                (self.baka[self.lowest])
                        ) |
                        (self.short[self.lowest] | self.tall[self.highest] | self.tall[self.higher]) |
                        (self.lewd[self.high] | self.lewd[self.higher] | self.lewd[self.highest]) |
                        (self.s[self.high] | self.s[self.higher] | self.s[self.highest]) |
                        (self.flashy[self.high] | self.flashy[self.higher]) |
                        (self.innocent[self.lowest]) |
                        (
                                (self.female[self.high] | self.female[self.higher] | self.female[self.highest]) &
                                (
                                        (self.tsundere[self.lowest]) &
                                        (self.catgirl[self.lowest]) &
                                        (self.skilled[self.highest]) &
                                        (
                                                (self.busty[self.higher] | self.busty[self.highest]) |
                                                (self.flat[self.low])
                                        ) &
                                        (self.yandere[self.lowest]) |
                                        (self.cheerful[self.lower] & self.gloomy[self.higher])
                                )
                        )
                )
        ), self.likeness['high'])

    def is_waifu(self):
        return ctrl.Rule(super().is_waifu() & (
                (
                        (self.kawaii[self.lowest]) |
                        (self.beautiful[self.high] | self.beautiful[self.higher] | self.beautiful[self.highest])
                ) &
                (
                        (
                                (self.smart[self.high] | self.smart[self.highest] | self.smart[self.higher]) &
                                (self.baka[self.lowest])
                        ) |
                        (self.tall[self.high] | self.tall[self.highest] | self.tall[self.higher]) |
                        (self.innocent[self.low]) |
                        (self.flashy[self.high]) |
                        (self.lewd[self.high] | self.lewd[self.higher] | self.lewd[self.highest]) |
                        (self.s[self.high] | self.s[self.higher] | self.s[self.highest]) |
                        (
                                (self.female[self.high] | self.female[self.higher] | self.female[self.highest]) &
                                (
                                        (self.tsundere[self.lower]) |
                                        (self.busty[self.high] | self.busty[self.higher]) |
                                        (self.cheerful[self.lower] & self.gloomy[self.high]) |
                                        (self.yandere[self.lowest])
                                )
                        )
                )
        ), self.likeness['average'])

    def is_trash(self):
        return ctrl.Rule(super().is_trash() & (
                (
                    (self.kawaii[self.highest] | self.kawaii[self.higher])
                ) |
                (
                        (self.baka[self.highest] | self.baka[self.higher]) &
                        (self.smart[self.lowest])
                ) |
                (self.short[self.high] | self.tall[self.higher] | self.tall[self.highest]) |
                (self.lewd[self.lowest]) |
                (self.plain[self.highest] | self.flashy[self.higher]) |
                (self.male[self.high] | self.male[self.higher] | self.male[self.highest]) &
                (
                        (self.female[self.high] | self.female[self.higher] | self.female[self.highest]) &
                        (
                                (self.m[self.high] | self.m[self.higher] | self.m[self.highest]) |
                                (self.flat[self.lowest]) |
                                (self.flashy[self.lowest] | self.flashy[self.lower] | self.flashy[self.low]) |
                                (self.yandere[self.high] | self.yandere[self.higher] | self.yandere[self.highest]) |
                                (self.cheerful[self.higher] | self.cheerful[self.highest])
                        )
                )
        ), self.likeness['low'])


class LewdAmazonLover2(LewdAmazonLover):
    def __init__(self):
        super().__init__()

        def default_setup_fun(trait):
            r = np.sort(np.random.choice(self[trait].universe, len(self._labels) * 4))

            def _stp_fun():
                self[trait][self.lowest] = fuzz.trapmf(self[trait].universe, r[:4])
                self[trait][self.lower] = fuzz.trapmf(self[trait].universe, r[4:8])
                self[trait][self.low] = fuzz.trapmf(self[trait].universe, r[8:12])
                self[trait][self.high] = fuzz.trapmf(self[trait].universe, r[12:16])
                self[trait][self.higher] = fuzz.trapmf(self[trait].universe, r[16:20])
                self[trait][self.highest] = fuzz.trapmf(self[trait].universe, r[20:])

            return _stp_fun

        self.lowest = 'E'
        self.lower = 'D'
        self.low = 'C'
        self.high = 'B'
        self.higher = 'A'
        self.highest = 'EX'

        self._labels = [self.lowest, self.lower, self.low, self.high, self.higher, self.highest]

        for trait in self._traits:
            self["setup_{}".format(trait)] = default_setup_fun(trait)


class LewdAmazonLover3(LewdAmazonLover):
    def __init__(self):
        super().__init__()

        def default_setup_fun(trait):
            return lambda: self[trait].automf(6, names=self._labels)

        self.lowest = 'E'
        self.lower = 'D'
        self.low = 'C'
        self.high = 'B'
        self.higher = 'A'
        self.highest = 'EX'

        self._labels = [self.lowest, self.lower, self.low, self.high, self.higher, self.highest]

        self.likeness['low'] = fuzz.zmf(self.likeness.universe, 2, 4)
        self.likeness['average'] = fuzz.gbellmf(self.likeness.universe, 2, 2, 6)
        self.likeness['high'] = fuzz.sigmf(self.likeness.universe, 8, 2)

        for trait in self._traits:
            self["setup_{}".format(trait)] = default_setup_fun(trait)
