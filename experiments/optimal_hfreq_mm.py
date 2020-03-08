"""
Recreation of Optimal High-Frequency Market Making by Takahiro Fushimi, Christian Gonz´alez Rojas,and Molly Herman
"""

from alpha_vantage.timeseries import TimeSeries

sym = "AAPL"

with open("APIKEY.txt", "r") as f:
    key = f.readline()

ts = TimeSeries(key=key)

data, meta_data = ts.get_intraday(sym, interval="1min")
print(type(data))

