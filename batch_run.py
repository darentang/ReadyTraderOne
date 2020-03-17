import json
from collections import namedtuple
import os
import subprocess
import time

def send_command(command, timer=None):
    p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    (output, err) = p.communicate()
    print(output)
    p_status = p.wait()

working_directory = os.getcwd()
directory = "/home/darentang/projects/readytraderone/ready_trader_one/"

Session = namedtuple('Session', ['path', 'day', 'bots', 'speed'])

bots = {}

for file in os.listdir("./ready_trader_one"):
    if "json" in file and "exchange" not in file:
        with open("./ready_trader_one/" + file, "r") as f:
            config = json.loads(f.read())
        bots[config["TeamName"]] = file[:-5]


template_file = "exchange_template.json"

with open(template_file, "r") as f:
    exchange = json.loads(f.read())

speed = 1.0

sessions = []
for day in range(1, 11):
    sessions.append(Session("all", str(day), ["Tradies", "SusumBot", "LilAkuma", "NowUCMe"], speed))

# for day in range(1, 11):
#     sessions.append(Session("susum", str(day), ["SusumBot", "DynamicInventory"], speed))

# for day in range(1, 11):
#     sessions.append(Session("alec", str(day), ["AlecBotV2", "DynamicInventory"], speed))

# for day in range(1, 11):
#     sessions.append(Session("fusion", str(day), ["FusionBot", "DynamicInventory"], speed))



with open("run_template.txt", "r") as f:
    run_template = f.read()

for i, session in enumerate(sessions):
    run_file = run_template
    path = f"/home/darentang/projects/readytraderone/runs/{session.path}/Day{session.day}/"
    data_file = path + "data.csv"
    if not os.path.exists(path):
        os.makedirs(path)
    exchange["Engine"]["MarketDataFile"] = directory + f"data/day{session.day}.csv"
    exchange["Engine"]["MatchEventsFile"] = data_file
    exchange["Engine"]["Speed"] = session.speed
    
    run_file = run_file.replace("#BOTNAMES#", str([bots[x] for x in session.bots]) )

    with open(directory + "run.py", "w") as f:
        f.write(run_file)
    
    with open(directory + "exchange.json", "w") as f:
        f.write(json.dumps(exchange, indent=4))

    with open(path + "details.json", "w") as f:
        f.write(json.dumps(session._asdict(), indent=4))

    print(f"Running session {i + 1}/{len(sessions)}.")
    send_command("cd /home/darentang/projects/readytraderone/ready_trader_one/ \n python3 run.py")

    for participant in session.bots:
        log_file = directory + bots[participant] + '.log'
        send_command(f"cp {log_file} {path+participant}.log")

    send_command(f"rm {directory}*.log")
        

    send_command(f"python3 {working_directory}/liveplot.py {data_file} {path + 'overall.png'}")
    send_command(f"python3 {working_directory}/benchmark.py {data_file} {path + 'analysis.json'}")
