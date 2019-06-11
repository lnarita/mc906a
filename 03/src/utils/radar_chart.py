# Libraries
from math import pi

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.waifus.monogatari import Senjougahara


def create_data_frame_from_waifu(waifu):
    data = {'group': [waifu.name]}
    for key, value in waifu.items():
        if key != 'name' and value != 0:
            data[key] = [value]
    return data


# ------- PART 1: Define a function that do a plot for one line of the dataset!
def make_spider(waifu, color):
    df = pd.DataFrame(create_data_frame_from_waifu(waifu))
    # number of variable
    categories = list(df)[1:]
    N = len(categories)

    # We are going to plot the first line of the data frame.
    # But we need to repeat the first value to close the circular graph:
    values = df.loc[0].drop('group').values.flatten().tolist()
    values += values[:1]

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)

    plt.title(waifu.name, size=11, color=color, y=1.1)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, color='grey', size=8)
    # Draw ylabels
    ax.set_rlabel_position(90)
    plt.yticks(np.arange(0, 11, 1), color=color, size=7)
    plt.ylim(0, 10)

    # Plot data
    ax.plot(angles, values, linewidth=1, linestyle='solid')

    # Fill area
    ax.fill(angles, values, 'b', alpha=0.1)


if __name__ == '__main__':
    make_spider(Senjougahara(), "grey")
    plt.show()
