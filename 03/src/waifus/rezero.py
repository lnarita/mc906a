from singleton_decorator import singleton

from src.waifu import BaseWaifu


@singleton
class Emilia(BaseWaifu):

    def __init__(self):
        super().__init__()
        self.name = "Emilia"
        self.beautiful = 6
        self.slim = 7
        self.smart = 4
        self.short = 3
        self.tall = 4
        self.innocent = 5
        self.female = 10
        self.skilled = 6
        self.gloomy = 7
        self.plain = 8
        self.busty = 5


@singleton
class Rem(BaseWaifu):

    def __init__(self):
        super().__init__()
        self.name = "Rem"
        self.kawaii = 8
        self.beautiful = 9
        self.slim = 7
        self.baka = 5
        self.short = 3
        self.tall = 4
        self.poor = 3
        self.innocent = 3
        self.lewd = 6
        self.female = 10
        self.s = 6
        self.skilled = 6
        self.cheerful = 9
        self.busty = 6
        self.yandere = 2


@singleton
class Felix(BaseWaifu):

    def __init__(self):
        super().__init__()
        self.name = "Felix Argyle"
        self.kawaii = 9
        self.beautiful = 6
        self.slim = 9
        self.baka = 10
        self.short = 6
        self.innocent = 7
        self.lewd = 7
        self.female = 4
        self.male = 6
        self.clumsy = 6
        self.cheerful = 8
        self.flashy = 8
        self.flat = 10
        self.megane = 3
        self.catgirl = 7
        self.trap = 10
        self.maid = 5
        self.butler = 5
