from singleton_decorator import singleton

from src.waifu import BaseWaifu


@singleton
class Astolfo(BaseWaifu):
    __instance = None

    def __init__(self):
        super().__init__()
        self.male = 5
        self.female = 7
        self.trap = 10
        self.baka = 9
        self.cheerful = 9
        self.flat = 10
