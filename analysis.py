import os
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
dfs = [pd.read_csv("ready_trader_one/match_events.csv"), pd.read_csv("logs/round2/match31_events.csv")]
names = ["emulated", "actual"]


teams = ["TeamJ"]




for team in teams:
    path = "img/" + team

    if not os.path.exists(path):
        os.mkdir(path)
    
    for (df, name) in zip(dfs, names):
        for side in ["B", "S"]:
            data = df.query(f"Competitor == '{team}'")
            inserts = data.query(f"Operation == 'Insert' and  FuturePrice <= Price <= EtfPrice and Side == '{side}'")
            # inserts = data.query(f"Operation == 'Insert' and  EtfPrice <= Price and Side == '{side}'")
            inserts = data.query(f"Operation == 'Insert' and  Price >= FuturePrice and Side == '{side}'")
            
            etf_prices = inserts["EtfPrice"]
            etf_position = inserts["EtfPosition"]
            future_prices = inserts["FuturePrice"]
            insert_prices = inserts["Price"]

            price_to_etf_diff = (insert_prices - etf_prices).to_numpy()
            future_to_etf_diff = (future_prices - etf_prices).to_numpy()

            # rows = np.unique(price_to_etf_diff)
            # cols = np.unique(future_to_etf_diff)

            # indices = np.meshgrid(rows, cols)

            # counts = np.zeros((len(cols), len(rows)))

            # for i in range(len(rows)):
            #     for j in range(len(cols)):
            #         r = indices[0][j][i]
            #         c = indices[1][j][i]

            #         counts[j][i] += np.sum((price_to_etf_diff == r) * (future_to_etf_diff == c) )

            # counts = np.flip(counts, axis=0)
            # counts

            # y = indices[0].flatten()
            # x = indices[1].flatten()

            plt.scatter(etf_position, future_to_etf_diff, label=name+side, alpha=0.2)


            # plt.scatter(x, y, s=counts)

plt.legend()
plt.show()