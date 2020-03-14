import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys

if len(sys.argv) == 1:
    filename = "ready_trader_one/match_events.csv"
elif len(sys.argv) == 2:
    filename = sys.argv[1]
else:
    filename = sys.argv[1]
    output = sys.argv[2]

competitors = None

if len(sys.argv) > 3:
    competitors = sys.argv[3:]
    

fig = plt.figure(figsize=(20,20))
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)

def animate(i):
    global competitors
    ax1.clear()
    ax2.clear()
    ax3.clear()
    df = pd.read_csv(filename)

    time = df["Time"]
    etf_price = df["EtfPrice"].fillna(method="bfill")
    future_price = df["FuturePrice"].fillna(method="bfill")
    if competitors is None:
        competitors = df["Competitor"].unique()
    for competitor in competitors:
        if competitor == "IgnoreMe":
            continue
        competitor_data = df.query(f"Competitor == '{competitor}'")
        profit_and_loss = competitor_data["ProfitLoss"]
        position = competitor_data["EtfPosition"]
        t = competitor_data["Time"]
        ax1.plot(t, profit_and_loss, label=competitor)
        ax2.plot(t, position, label=competitor)
    ax1.legend(loc=2)
    ax1.set_title("Profit/Loss")
    
    ax2.legend(loc=2)
    ax2.set_title("Position")

    ax3.plot(time, etf_price, label="ETF Last Traded Price")
    ax3.plot(time, future_price, label="Future Last Traded Price")
    ax3.legend(loc=2)
    ax3.set_title("Prices")

if len(sys.argv) == 1 or len(sys.argv) == 2:
    A = animation.FuncAnimation(fig, animate, interval=1000)
    plt.show()
elif len(sys.argv) >= 3:
    animate(0)
    plt.savefig(output)
