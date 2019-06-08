import sys
import time
import random
sys.path.insert(0, '/Users/williamcruz/warzone/mc906a/03/src')

import matplotlib
matplotlib.use("TKAgg")

from tkinter import *
from profiles.first import *
from waifus.monogatari import *
from waifus.myheroacademia import *
from recommender import Recommender

WIDTH = 480
HEIGHT = 720
WAIFUS = [Shinobu(), Senjougahara(), Hanekawa(), Hachikuji(), Sengoku(), Kanbaru()]

class DemoTinderWindow():
	def __init__(self, profile):
		self.profile = profile

	def should_recomend_waifu(self, waifu):
		######## TODO Right way of actually recommend waifus ##########
		print("RECOMENDER PROBABILITY CALCULATOR: ", Recommender.calculate_like_probability(self.profile, waifu))
		return Recommender.calculate_like_probability(self.profile, waifu) > 4.0

	def get_waifu_card(self, waifu):
		card = self.canvas.create_rectangle(0, 0, 480, 720, fill="#CACACA")
		label = self.canvas.create_text((200, 600), text=waifu.__class__.__name__)
		return (card, label)

	def main_loop(self):
		window = Tk()
		window.title("Tinder app")
		window.geometry('{}x{}'.format(WIDTH, HEIGHT))

		self.canvas = Canvas(window, width=WIDTH, height=HEIGHT)
		self.canvas.pack()

		cards = []
		for waifu in WAIFUS:
			(card, label) = self.get_waifu_card(waifu)
			is_recommended = self.should_recomend_waifu(waifu)
			window.update()
			time.sleep(2)

			new_position = 10 if is_recommended else -10

			for x in range(0, int(WIDTH/10)):
				self.canvas.move(card, new_position, 0)
				self.canvas.move(label, new_position, 0)
				window.update()
				time.sleep(.01)

		self.canvas.tag_lower(card)
		self.canvas.tag_lower(label)

		window.mainloop()



# Running...
DemoTinderWindow(ShortStacksLover()).main_loop()