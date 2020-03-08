from alpha_vantage.timeseries import TimeSeries
import json

sym = "AAPL"
day = "2017-06-12"

with open("APIKEY.txt", "r") as f:
    key = f.readline()

ts = TimeSeries(key=key, output_format="json")

data, meta_data = ts.get_intraday(sym, interval="1min", outputsize="full")

with open(f"data/{sym}_data.json", "w") as f:
    json.dump(data, f)

print(data.keys())