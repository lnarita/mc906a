from singleton_decorator import singleton

from src.waifu import BaseWaifu


@singleton
class Chitoge(BaseWaifu):
    __instance = None

    def __init__(self):
        super().__init__()
        self.beautiful = 7
        self.slim = 7
        self.baka = 8
        self.tall = 6
        self.rich = 5
        self.lewd = 5
        self.female = 10
        self.s = 6
        self.clumsy = 6
        self.cheerful = 8
        self.flashy = 9
        self.flat = 5
        self.tsundere = 10

@singleton
class Kosaki(BaseWaifu):
    __instance = None

    def __init__(self):
        super().__init__()
        self.kawaii = 8
        self.beautiful = 6
        self.slim = 7
        self.smart = 3
        self.baka = 7
        self.short = 4
        self.rich = 3
        self.innocent = 10
        self.female = 10
        self.clumsy = 9
        self.cheerful = 6
        self.plain = 6
        self.flat = 3
        self.megane = 2


@singleton
class Tsugumi(BaseWaifu):
    __instance = None

    def __init__(self):
        super().__init__()
        self.beautiful = 6
        self.chubby = 3
        self.baka = 8
        self.tall = 7
        self.rich = 2
        self.poor = 2
        self.innocent = 7
        self.lewd = 6
        self.female = 10
        self.m = 4
        self.clumsy = 4
        self.cheerful = 4
        self.gloomy = 6
        self.flashy = 3
        self.plain = 4
        self.busty = 8
        self.megane = 5
        self.tomboy = 10
        self.tsundere = 6


@singleton
class Marika(BaseWaifu):
    __instance = None

    def __init__(self):
        super().__init__()
        self.beautiful = 7
        self.slim = 8
        self.smart = 6
        self.short = 3
        self.tall = 3
        self.rich = 10
        self.lewd = 9
        self.female = 10
        self.s = 2
        self.skilled = 5
        self.cheerful = 5
        self.flashy = 8
        self.flat = 4
        self.megane = 2
        self.maid = 2
        self.yandere = 2