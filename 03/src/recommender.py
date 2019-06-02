from src.config import TRAITS
from src.husbandos.Fate import Merlin
from src.profiles.Profile1 import Profile1


class Recommender:
    @staticmethod
    def calculate_like_probability(user, waifu):
        result = user.predict(waifu)
        for trait in TRAITS:
            user[trait].view()
        user.likeness.view()
        user.likeness.view(sim=result)
        return result.output['likeness']

