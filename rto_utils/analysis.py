import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def get_profit(events, teamname):
    """
    Return profit and time series
    """
    df = events.query(f"Competitor == '{teamname}'")
    return df["Time"].to_numpy(),  df["ProfitLoss"].to_numpy()
    


def gradient(events, teamname, window=50):
    """
    Output the gradient from fixed window least square fit
    """
    time, profit = get_profit(events, teamname)
    N = len(time)
    last = N - N % window

    assert N == len(profit)

    grad = np.zeros_like(time)
    ls = np.zeros_like(time)

    for i in range(0, N, window):
        start = i
        stop = start + window if start <= last else N

        p = profit[start:stop]
        t = time[start:stop]
        
        coeff = np.polyfit(t, p, 1)

        grad[start:stop] = coeff[0]
        ls[start:stop] = np.polyval(coeff, t)

    return grad, ls



def side_color(bots, name, side):
    rgb = bots[name]
    if side == "S":
        rgb = rgb + 200
        rgb = rgb / np.max((np.max(rgb) , 255))
    else:
        rgb = rgb / 255 

    return tuple(rgb)

def normalise(rgb):
    return np.abs(rgb) / np.max((np.max(rgb) , 255))


def invert(bots, name, side):
    rgb = bots[name]
    rgb = (255 - rgb) - 100

    return normalise(rgb)


def surgery(df, time_range, bots, outfile, nums=None):
    """
    Shows ordering activity in teams against time

    # Arguments
        df (pd dataframe)       : dataframe of the market events
        time_range (tuple)      : start and end time
        bots (dict)             : dictionary of the participants wished to be looked at. Dict values should be
                                  the colour of the particpant shown in the figure
        outfile (string)        : output file directory
        nums (str or None)      : "id" means order id to be shown, "volume" means order volumes to be shown

    """
    c = "b"
    t = 5

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
                ax.scatter(insert_time, insert_price, marker=marker, color=invert(bots, bot_name, side), zorder=2)
                if nums is not None:
                    if nums == "volume":
                        ax.text(insert_time + 0.02, insert_price - 0.02, str(int(insert_volume)))
                    elif nums == "id":
                        ax.text(insert_time + 0.02, insert_price - 0.02, str(int(order_id)))
                if len(cancel) == 1:
                    cancel_time = cancel["Time"].values[-1]
                    ax.plot([insert_time, cancel_time], [insert_price, insert_price], color=side_color(bots, bot_name, side), linewidth=t, zorder=1)
                    ax.scatter(cancel_time, insert_price, marker="x", color=invert(bots, bot_name, side), zorder=2)
                elif len(fill) > 0:
                    fill_time = fill["Time"].values[-1]
                    ax.plot([insert_time, fill_time], [insert_price, insert_price], color=side_color(bots, bot_name, side), linewidth=t, zorder=1)

                    for time in fill["Time"].values:
                        ax.scatter(time, insert_price, marker="*", color="r", zorder=2)


    ax.plot(bot_data["Time"], bot_data["EtfPrice"], color='orange')
    ax.plot(bot_data["Time"], bot_data["FuturePrice"], color='skyblue')

    ax.yaxis.set_ticks(np.arange(np.min(bot_data["EtfPrice"]) - 5, np.max(bot_data["EtfPrice"]) + 5, 1))
    ax.grid(axis='y')


    ax2.legend()
    ax2.yaxis.set_ticks([-100, -75, -50, -25, 0 ,25, 50, 75, 100])
    ax2.grid(axis='y')
    pnl.legend()

    legend_elements = []
    legend_elements.append(Line2D([0], [0], color='orange', label="ETF"))
    legend_elements.append(Line2D([0], [0], color='skyblue', label="Future"))
    for bot, color in bots.items():
        legend_elements.append(Line2D([0], [0], color=side_color(bots, bot, "B"), label=bot + " bids"))
        legend_elements.append(Line2D([0], [0], color=side_color(bots, bot, "S"), label=bot + " asks"))
    legend_elements.append(Line2D([0], [0], marker="x", color=c, label="Cancel"))
    legend_elements.append(Line2D([0], [0], marker="v", color=c, label="Buy"))
    legend_elements.append(Line2D([0], [0], marker="^", color=c, label="Sell"))
    legend_elements.append(Line2D([0], [0], marker="*", color=c, label="Fill"))


    ax.legend(handles=legend_elements, loc=0)

    plt.savefig(outfile)


