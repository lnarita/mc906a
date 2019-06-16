from singleton_decorator import singleton

from src.waifu import BaseWaifu


@singleton
class Asuka(BaseWaifu):

    def __init__(self):
        super().__init__()
        self.name = "Asuka Langley Soryu"
        self.beautiful = 5
        self.slim = 8
        self.smart = 7
        self.tall = 5
        self.rich = 3
        self.poor = 2
        self.lewd = 6
        self.female = 10
        self.s = 8
        self.skilled = 5
        self.gloomy = 6
        self.flashy = 7
        self.flat = 5
        self.tsundere = 10


@singleton
class Rei(BaseWaifu):

    def __init__(self):
        super().__init__()
        self.name = "Rei Ayanami"
        self.kawaii = 3
        self.beautiful = 4
        self.slim = 7
        self.smart = 2
        self.baka = 7
        self.tall = 3
        self.innocent = 6
        self.female = 10
        self.skilled = 8
        self.gloomy = 9
        self.plain = 10
        self.flat = 3
