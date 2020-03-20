import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
df = pd.read_csv("match31_events.csv")

teams = ["TeaMaster"]

for team in teams:
    data = df.query(f"Competitor == '{team}'")
    inserts = data.query("Operation == 'Insert'")

    etf_prices = inserts["EtfPrice"]
    future_prices = inserts["FuturePrice"]
    insert_prices = inserts["Price"]

    price_to_etf_diff = (insert_prices - etf_prices).to_numpy()
    future_to_etf_diff = (future_prices - etf_prices).to_numpy()

    rows = np.unique(price_to_etf_diff)
    cols = np.unique(future_to_etf_diff)

    indices = np.meshgrid(rows, cols)

    counts = np.zeros((len(rows), len(cols)))

    for i in range(len(rows)):
        for j in range(len(cols)):
            r = indices[0][j][i]
            c = indices[1][j][i]

            counts[i][j] += np.sum((price_to_etf_diff == r) * (future_to_etf_diff == c) )

    counts = np.flip(counts, axis=0)
    x = indices[0].flatten()
    y = indices[1].flatten()


    plt.scatter(x, y, s=counts)

plt.show()