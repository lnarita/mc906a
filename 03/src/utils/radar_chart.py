# Libraries
from math import pi

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.profiles.second import *
from src.recommender import Recommender
from src.waifus.monogatari import Senjougahara


def create_data_frame_from_waifu(waifu):
    data = {'group': [waifu.name]}
    for key, value in waifu.items():
        if key != 'name' and value != 0:
            data[key] = [value]
    return data


def create_data_frame_from_user(user, mf):
    prefix = '{} - {}'.format(user.__class__.__name__, mf)
    data = {'group': ['{} max'.format(prefix), '{} min'.format(prefix)]}
    for trait in user._traits:
        term = user[trait].terms[mf]
        vmin = None
        vmax = None
        for idx, value in enumerate(term.mf):
            if value != 0 and not vmin:
                vmin = idx - 1
            elif vmin and value == 0:
                vmax = idx
                break

        if vmin is not None and vmax is None:
            vmax = 10
        vmin = vmin if vmin and vmin > -1 else 0
        vmax = vmax if vmax and vmax > -1 else 0
        if vmin is not None and vmax is not None:
            data[trait] = [vmax, vmin]
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

    plt.title(waifu.name, size=11, color='grey', y=1.1)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, color='grey', size=8)
    # Draw ylabels
    ax.set_rlabel_position(90)
    plt.yticks(np.arange(0, 11, 1), color='grey', size=7)
    plt.ylim(0, 10)

    # Plot data
    ax.plot(angles, values, linewidth=1, linestyle='solid', color=color)

    # Fill area
    ax.fill(angles, values, 'b', alpha=0.1)


def make_user_spider(profile, mf, color):
    df = pd.DataFrame(create_data_frame_from_user(profile, mf))
    # number of variable
    categories = list(df)[1:]
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)

    plt.title('{}'.format(mf), size=11, color='grey', y=1.1)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, color='grey', size=8)
    # Draw ylabels
    ax.set_rlabel_position(90)
    plt.yticks(np.arange(0, 11, 1), color='grey', size=7)
    plt.ylim(0, 10)

    # We are going to plot the first line of the data frame.
    # But we need to repeat the first value to close the circular graph:
    values1 = df.loc[0].drop('group').values.flatten().tolist()
    values1 += values1[:1]

    ax.plot(angles, values1, linewidth=1, linestyle='solid', color=color)

    values2 = df.loc[1].drop('group').values.flatten().tolist()
    values2 += values2[:1]

    ax.plot(angles, values2, linewidth=1, linestyle='solid', color=color)

    # Fill area
    ax.fill_between(angles, values1, values2, alpha=.2, color=color)


if __name__ == '__main__':
    user = LewdAmazonLover()
    waifu = Senjougahara()
    Recommender.calculate_like_probability(user, waifu, False)
    for mf in [user.lowest, user.lower, user.low, user.high, user.higher, user.highest]:
        create_data_frame_from_user(user, mf)
        make_user_spider(user, mf, 'darkorchid')
        # make_spider(waifu, "grey")
        plt.show()
        plt.cla()
