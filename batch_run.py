import json
from collections import namedtuple
import os

cwd = os.getcwd()


Session = namedtuple('Session', ['day', 'bots', 'speed'])

bots = {}

for file in os.listdir("./ready_trader_one"):
    if "json" in file and "exchange" not in file:
        with open("./ready_trader_one/" + file, "r") as f:
            config = json.loads(f.read())
        bots[config["TeamName"]] = file[:-5]

template_file = "exchange_template.json"

with open(template_file, "r") as f:
    exchange = json.loads(f.read())

sessions = [
    Session("1", ["SusumBot", "AlecBotV2", "FusionBot", "DynamicInventory"], 1.0)
]

print(bots)

with open("run_template.txt", "r") as f:
    run_template = f.read()

for i, session in enumerate(sessions):
    run_file = run_template
    path = f"./runs/{i}/"
    if not os.path.exists(path):
        os.mkdir(path)
    exchange["Engine"]["MarketDataFile"] = f"../ready_trader_one/data/day{session.day}.csv"
    exchange["Engine"]["Speed"] = session.speed
    
    run_file = run_file.replace("#BOTNAMES#", str([bots[x] for x in session.bots]) )

    with open(path + "run.py", "w") as f:
        f.write(run_file)