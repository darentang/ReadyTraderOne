import sys
sys.path.insert(0, "/home/darentang/projects/readytraderone")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rto_utils.analysis import surgery, get_profit, gradient

teams = {
    "TeamJ": np.array([150, 0, 0]), 
    "NPTrivial": np.array([0, 0, 150]), 
    # "Tradies": np.array([0, 150, 0]), 
    }

events_file = "match31_events.csv"
outpath = "img/analysis/"

# points of interest (start, end)
POI = [
    (306, 500),
    (1135, 1322),
    (1728, 1956)
]

events = pd.read_csv(events_file)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 20))
for team in teams.keys():
    grad, ls = gradient(events, team, window=300)
    time, profit = get_profit(events, team)

    ax1.plot(time, profit, label=team)
    ax1.plot(time, ls, label=team+"LS")

    ax2.plot(time, grad, label=team)

ax1.legend()
ax2.legend()
plt.savefig(outpath + "gradient.png")

for time_range in POI:
    surgery(events, time_range, teams, outpath + str(time_range[0]) + "_" + str(time_range[1]) + ".pdf")