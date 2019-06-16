import sys
import time
import random

import matplotlib
import numpy as np

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
		likeness = Recommender.calculate_like_probability(self.profile, waifu)
		print("RECOMENDER PROBABILITY CALCULATOR: ", likeness)
		trash = fuzz.interp_membership(np.arange(0, 11, 1), self.profile.likeness['low'].mf, likeness)
		waifu = fuzz.interp_membership(np.arange(0, 11, 1), self.profile.likeness['average'].mf, likeness)
		best_gril = fuzz.interp_membership(np.arange(0, 11, 1), self.profile.likeness['high'].mf, likeness)
		if best_gril > waifu and best_gril > trash:
			return 'B'
		if waifu > best_gril and waifu > trash:
			return 'W'
		if trash > waifu and trash > best_gril:
			return 'T'
		if likeness > 4.0:
			return 'W'
		else:
			return 'T'

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

		def _swipe_up():
			for y in range(0, int(HEIGHT/10)):
				self.canvas.move(card, 0, -10)
				self.canvas.move(label, 0, -10)
				window.update()
				time.sleep(.01)

		def _swipe_right():
			for x in range(0, int(WIDTH/10)):
				self.canvas.move(card, 10, 0)
				self.canvas.move(label, 10, 0)
				window.update()
				time.sleep(.01)

		def _swipe_left():
			for x in range(0, int(WIDTH/10)):
				self.canvas.move(card, -10, 0)
				self.canvas.move(label, -10, 0)
				window.update()
				time.sleep(.01)

		cards = []
		for waifu in WAIFUS:
			(card, label) = self.get_waifu_card(waifu)
			category = self.should_recomend_waifu(waifu)
			swipe = {
				'B': lambda: _swipe_up(),
				'W': lambda: _swipe_right(),
				'T': lambda: _swipe_left()
			}[self.should_recomend_waifu(waifu)]
			window.update()
			time.sleep(2)

			swipe()

		self.canvas.tag_lower(card)
		self.canvas.tag_lower(label)

		window.mainloop()


if __name__ == '__main__':
	DemoTinderWindow(ShortStacksLover()).main_loop()