from singleton_decorator import singleton

from src.waifu import BaseWaifu


@singleton
class Senjougahara(BaseWaifu):

    def __init__(self):
        super().__init__()
        self.name = "Hitagi Senjougahara"
        self.beautiful = 7
        self.slim = 10
        self.tall = 4
        self.poor = 5
        self.lewd = 8
        self.busty = 5
        self.female = 10
        self.s = 8
        self.gloomy = 4
        self.plain = 5
        self.tsundere = 9
        self.yandere = 6


@singleton
class Shinobu(BaseWaifu):

    def __init__(self):
        super().__init__()
        self.name = "Shinobu Oshino"
        self.kawaii = 5
        self.slim = 6
        self.baka = 9
        self.short = 7
        self.flat = 10
        self.innocent = 7
        self.female = 10
        self.s = 9
        self.plain = 6
        self.tsundere = 5


@singleton
class Hanekawa(BaseWaifu):

    def __init__(self):
        super().__init__()
        self.name = "Tsubasa Hanekawa"
        self.beautiful = 8
        self.slim = 6
        self.chubby = 4
        self.short = 6
        self.poor = 9
        self.innocent = 2
        self.lewd = 7
        self.skilled = 7
        self.megane = 10
        self.catgirl = 10
        self.smart = 10
        self.cheerful = 7
        self.busty = 9
        self.female = 10


@singleton
class Hachikuji(BaseWaifu):
    def __init__(self):
        super().__init__()
        self.name = "Mayoi Hachikuji"
        self.female = 10
        self.baka = 9
        self.cheerful = 9
        self.flat = 10
        self.kawaii = 7
        self.slim = 6
        self.innocent = 8
        self.clumsy = 8
        self.flashy = 5
        self.short = 8


@singleton
class Kanbaru(BaseWaifu):

    def __init__(self):
        super().__init__()
        self.name = "Suruga Kanbaru"
        self.female = 10
        self.baka = 7
        self.cheerful = 10
        self.flat = 6
        self.busty = 3
        self.beautiful = 3
        self.tomboy = 9
        self.slim = 6
        self.short = 4
        self.rich = 7
        self.lewd = 9
        self.m = 6
        self.clumsy = 7
        self.flashy = 8
        self.busty = 5


@singleton
class Sengoku(BaseWaifu):

    def __init__(self):
        super().__init__()
        self.name = "Nadeko Sengoku"
        self.female = 10
        self.baka = 7
        self.gloomy = 9
        self.flat = 10
        self.kawaii = 2
        self.slim = 6
        self.rich = 5
        self.clumsy = 7
        self.plain = 10
