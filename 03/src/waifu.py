from src.commons import Map
from src.config import TRAITS


class Waifu(Map):
    def __init__(self, traits):
        super().__init__()

        def add_trait(trait):
            self[trait] = 0

        for trait in traits:
            add_trait(trait)


class BaseWaifu(Waifu):
    def __init__(self):
        super().__init__(TRAITS)
