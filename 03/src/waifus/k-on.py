from singleton_decorator import singleton

from src.waifu import BaseWaifu


@singleton
class Yui(BaseWaifu):
    __instance = None

    def __init__(self):
        super().__init__()
        self.kawaii = 8
        self.beautiful = 5
        self.slim = 7
        self.baka = 9
        self.short = 5
        self.poor = 3
        self.innocent = 7
        self.female = 10
        self.clumsy = 9
        self.cheerful = 10
        self.flashy = 8
        self.flat = 4

@singleton
class Ritsu(BaseWaifu):
    __instance = None

    def __init__(self):
        super().__init__()
        self.kawaii = 5
        self.beautiful = 3
        self.slim = 3
        self.baka = 7
        self.short = 3
        self.poor = 2
        self.innocent = 3
        self.lewd = 4
        self.female = 10
        self.clumsy = 7
        self.skilled = 2
        self.cheerful = 8
        self.flashy = 7
        self.flat = 3
        self.busty = 3

@singleton
class Mio(BaseWaifu):
    __instance = None

    def __init__(self):
        super().__init__()
        self.kawaii = 8
        self.beautiful = 10
        self.slim = 8
        self.smart = 9
        self.tall = 6
        self.rich = 3
        self.innocent = 9
        self.female = 10
        self.clumsy = 3
        self.skilled = 5
        self.gloomy = 5
        self.busty = 5
        self.megane = 3
        self.catgirl = 10

@singleton
class Tsumugi(BaseWaifu):
    __instance = None

    def __init__(self):
        super().__init__()
        self.kawaii = 8
        self.slim = 6
        self.chubby = 3
        self.smart = 5
        self.baka = 5
        self.tall = 3
        self.rich = 9
        self.innocent = 2
        self.lewd = 5
        self.female = 10
        self.skilled = 6
        self.cheerful = 4
        self.gloomy = 4
        self.plain = 5
        self.flat = 3
        self.megane = 3

@singleton
class Azusa(BaseWaifu):
    __instance = None

    def __init__(self):
        super().__init__()
        self.kawaii = 2
        self.beautiful = 3
        self.slim = 6
        self.baka = 6
        self.short = 9
        self.innocent = 3
        self.female = 9
        self.clumsy = 7
        self.cheerful = 3
        self.gloomy = 7
        self.plain = 10
        self.flat = 10
        self.catgirl = 1
        self.tsundere = 3