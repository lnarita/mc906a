from singleton_decorator import singleton

from src.waifu import BaseWaifu


@singleton
class Uraraka(BaseWaifu):
    __instance = None

    def __init__(self):
        super().__init__()
        self.kawaii = 9
        self.beautiful = 9
        self.slim = 8
        self.smart = 6
        self.baka = 3
        self.short = 3
        self.poor = 6
        self.innocent = 7
        self.female = 10
        self.clumsy = 6
        self.cheerful = 9
        self.flashy = 6
        self.busty = 6


@singleton
class Tsuyu(BaseWaifu):
    __instance = None

    def __init__(self):
        super().__init__()
        self.kawaii = 7
        self.beautiful = 4
        self.slim = 8
        self.smart = 8
        self.short = 7
        self.innocent = 3
        self.female = 10
        self.skilled = 9
        self.gloomy = 5
        self.flat = 6


@singleton
class Toga(BaseWaifu):
    __instance = None

    def __init__(self):
        super().__init__()
        self.kawaii = 2
        self.beautiful = 5
        self.slim = 4
        self.smart = 4
        self.lewd = 9
        self.female = 10
        self.s = 9
        self.skilled = 8
        self.cheerful = 7
        self.gloomy = 5
        self.flashy = 8
        self.busty = 5
        self.yandere = 9


@singleton
class Momo(BaseWaifu):
    __instance = None

    def __init__(self):
        super().__init__()
        self.kawaii = 5
        self.beautiful = 9
        self.slim = 6
        self.smart = 10
        self.tall = 6
        self.rich = 10
        self.innocent = 9
        self.female = 10
        self.clumsy = 4
        self.skilled = 7
        self.cheerful = 6
        self.busty = 9
        self.megane = 6


@singleton
class Jirou(BaseWaifu):
    __instance = None

    def __init__(self):
        super().__init__()
        self.kawaii = 6
        self.beautiful = 6
        self.slim = 6
        self.smart = 5
        self.short = 3
        self.rich = 4
        self.lewd = 5
        self.female = 10
        self.skilled = 7
        self.gloomy = 6
        self.flashy = 6
        self.busty = 4


@singleton
class Mina(BaseWaifu):
    __instance = None

    def __init__(self):
        super().__init__()
        self.kawaii = 6
        self.beautiful = 4
        self.slim = 4
        self.baka = 10
        self.short = 3
        self.innocent = 9
        self.female = 10
        self.clumsy = 8
        self.cheerful = 9
        self.flashy = 8
        self.busty = 5
