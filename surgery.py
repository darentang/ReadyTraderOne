import sys
import pandas as pd
import matplotlib.pyplot as plt
from collections import namedtuple
from matplotlib.lines import Line2D
import numpy as np


bots = {
    "Tradies": np.array([0, 150, 0]), 
    "TeamJ": np.array([150, 0, 0]),
    # "NowUCMe": np.array([0, 0, 150]),
    # "SMCtrading": np.array([0, 150, 150]),
    # "LilAkuma": np.array([150, 150, 0]),
    # "SMCtrading": np.array([150, 0, 150]),

    # "AlecBotV2": np.array([0, 0, 150]),
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
df = df.query(f"Time > {time_range[0]}").query(f"Time < {time_range[1]}")

fig = plt.figure(constrained_layout=True, figsize=(20, 10))
spec = fig.add_gridspec(ncols=1, nrows=3, height_ratios=[1, 1, 7])

ax2 = fig.add_subplot(spec[0, 0])
pnl = fig.add_subplot(spec[1, 0])
ax = fig.add_subplot(spec[2, 0])

for i, bot_name in enumerate(bots.keys()):
    bot_data = df.query(f"Competitor == '{bot_name}'")
    orders = bot_data["OrderId"].dropna().unique()
    ax2.plot(bot_data["Time"], bot_data["EtfPosition"], label=bot_name)
    pnl.plot(bot_data["Time"], bot_data["ProfitLoss"], label=bot_name)

    for order_id in orders:
        data = bot_data.query(f"OrderId == {order_id}")

        insert = data.query(f"Operation == 'Insert'")
        cancel = data.query(f"Operation == 'Cancel'")
        fill = data.query(f"Operation == 'Fill'")

        if len(insert) == 1:
            side = insert["Side"].values[0]
            insert_price = insert["Price"].values[0] - i * 0.2
            insert_time = insert["Time"].values[0]
            insert_volume = int(insert["Volume"].values[0])
            if side == "B":
                marker = "v"
            else:
                marker = "^"
            ax.scatter(insert_time, insert_price, marker=marker, color=invert(bot_name, side), zorder=2)
            ax.text(insert_time + 0.02, insert_price - 0.02, str(int(insert_volume)))
            # ax.text(insert_time + 0.02, insert_price - 0.02, str(int(order_id)))
            if len(cancel) == 1:
                cancel_time = cancel["Time"].values[-1]
                ax.plot([insert_time, cancel_time], [insert_price, insert_price], linestyle=linestyle[side], color=side_color(bot_name, side), linewidth=t, zorder=1)
                ax.scatter(cancel_time, insert_price, marker="x", color=c, zorder=2)
            elif len(fill) > 0:
                fill_time = fill["Time"].values[-1]
                ax.plot([insert_time, fill_time], [insert_price, insert_price], linestyle=linestyle[side], color=side_color(bot_name, side), linewidth=t, zorder=1)

                for time in fill["Time"].values:
                    ax.scatter(time, insert_price, marker="*", color=c, zorder=2)


ax.plot(bot_data["Time"], bot_data["EtfPrice"], color='orange')
ax.plot(bot_data["Time"], bot_data["FuturePrice"], color='skyblue')

ax.yaxis.set_ticks(np.arange(np.min(bot_data["EtfPrice"]) - 5, np.max(bot_data["EtfPrice"]) + 5, 1))
ax.grid(axis='y')


ax2.legend()
pnl.legend()

legend_elements = []
legend_elements.append(Line2D([0], [0], color='orange', label="ETF"))
legend_elements.append(Line2D([0], [0], color='skyblue', label="Future"))
for bot, color in bots.items():
    legend_elements.append(Line2D([0], [0], color=side_color(bot, "B"), label=bot + " bids"))
    legend_elements.append(Line2D([0], [0], color=side_color(bot, "S"), label=bot + " asks"))
legend_elements.append(Line2D([0], [0], marker="x", color=c, label="Cancel"))
legend_elements.append(Line2D([0], [0], marker="v", color=c, label="Insert"))
legend_elements.append(Line2D([0], [0], marker="*", color=c, label="Fill"))


ax.legend(handles=legend_elements, loc=0)

plt.savefig(outfile)
