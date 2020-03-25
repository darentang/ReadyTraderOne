import os
import sys
sys.path.insert(0, os.getcwd())

import pandas as pd
import matplotlib.pyplot as plt
from collections import namedtuple
from matplotlib.lines import Line2D

import numpy as np

from rto_utils.analysis import surgery


bots = {
    # "Tradies": np.array([0, 150, 0]), 
    # "TeamJ": np.array([150, 0, 0]),
    # "SusumBot": np.array([0, 150, 0]),
    # "TeaMaster": np.array([0, 150, 150]),
    "NPTrivial": np.array([0, 0, 150]),
}


linestyle = {
    "B": "-",
    "S": "-"
}

def side_color(name, side):
    rgb = bots[name]
    if side == "S":
        rgb = rgb + 200
        rgb = rgb / np.max((np.max(rgb) , 255))
    else:
        rgb = rgb / 255 

    return tuple(rgb)

def normalise(rgb):
    return rgb / np.max((np.max(rgb) , 255))


def invert(name, side):
    rgb = bots[name]
    rgb = 255 - rgb

    return normalise(rgb)

if len(sys.argv) >= 3:
    filename = sys.argv[1]
    outfile = sys.argv[2]
else:
    filename = "ready_trader_one/match_events.csv"
    outfile = "img/pricing2.pdf"

c = "b"
t = 5


if len(sys.argv) == 5:
    start_time = int(sys.argv[3])
    duration = int(sys.argv[4])
else:
    start_time = 250
    duration = 50

time_range = (start_time, start_time + duration)

df = pd.read_csv(filename)

surgery(df, time_range, bots, outfile, nums="id")