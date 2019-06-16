from src.config import TRAITS
from src.profiles.second import *
from src.waifus.monogatari import *


class Recommender:
    @staticmethod
    def calculate_like_probability(user, waifu, vis_membership_func=False):
        result = user.predict(waifu)
        if vis_membership_func:
            for trait in TRAITS:
                user[trait].view()
        # user.likeness.view()
        # user.likeness.view(sim=result)
        return result.output['likeness']


if __name__ == '__main__':
    user = LewdAmazonLover3()
    waifus = [Sengoku(), Kanbaru(), Hachikuji(), Hanekawa(), Shinobu(), Senjougahara()]
    print([Recommender.calculate_like_probability(user, waifu, waifu == Sengoku() and False) for waifu in waifus])
