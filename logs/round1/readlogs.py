import re
import numpy as np
filename = "match31_Tradies.log"
with open(filename, "r") as f:
    data = f.read()

def to_time(quote):
    return 3600.0 * float(quote[0]) + 60.0 * float(quote[1]) + float(quote[2]) + float(quote[3]) / 1000.0


template = r'.+\s(\d+):(\d+):(\d+),(\d+)\s.+\((\d+), (\d+)\). Bid: (\d+) @ \$(\d+), Ask: (\d+) @ \$(\d+)'
quotes = re.findall(template, data)

out_data = np.empty((len(quotes), 3))

for i, quote in enumerate(quotes):
    time = to_time(quote)
    best_bid, best_ask = quote[7], quote[9]
    out_data[i, :] = [time, best_bid, best_ask]

out_data = out_data[np.argsort(out_data[:, 0])]
out_data[:, 0] -= out_data[0, 0]
np.savetxt("best_ask_bid.csv", out_data, header="Time,bid,ask", delimiter=",")