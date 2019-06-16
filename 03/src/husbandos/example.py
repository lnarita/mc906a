from singleton_decorator import singleton

from src.waifu import BaseWaifu


@singleton
class ExampleHusbando(BaseWaifu):
    __instance = None

    def __init__(self):
        super().__init__()
        self.male = 10
        self.baka = 7
        self.smart = 9
        self.cheerful = 7
        self.gloomy = 4
        self.clumsy = 7
        self.skilled = 7
