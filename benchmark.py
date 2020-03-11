"""
Evaluation of models
"""
import pandas as pd
import numpy as np
import networkx as nx
import json
import sys
from networkx.drawing.nx_agraph import to_agraph 

import matplotlib.pyplot as plt
from collections import defaultdict

if len(sys.argv) == 1:
    filename = "ready_trader_one/match_events.csv"
elif len(sys.argv) == 2:
    filename = sys.argv[1]
    output = "analysis.json"
else:
    filename = sys.argv[1]
    output = sys.argv[2]

df = pd.read_csv(filename)
bots = df["Competitor"]

# stores dataframe for each competitor
competitors = dict()

for bot in bots.unique():
    competitors[bot] = dict(events=df.query(f"Competitor == '{bot}'"))

def get_profit_loss(competitors, info):
    for name, data in competitors.items():
        profit_loss = data['events']['ProfitLoss']
        info[name]['PnL'] = dict(last=np.round(profit_loss.iloc[-1], 2), avg=np.round(np.mean(profit_loss), 2), std=np.round(np.std(profit_loss), 2))

def get_liquidity(competitors, info, size=1):
    for name, data in competitors.items():
        orders = data['events']['OrderId'].dropna().unique()
        cancel_rate = 0
        fill_rate = 0
        fill_time = []
        print(f"Calculating {name} with {size} * {len(orders)} data points")
        for order in np.random.choice(orders, size=int(size * len(orders)), replace=False):
            evt = data['events'].query(f"OrderId == '{order}'")
            insert = evt.query(f"Operation == 'Insert'")
            fill = evt.query(f"Operation == 'Fill'")
            cancel = evt.query(f"Operation == 'Cancel'")

            if len(fill) > 0:
                fill_rate += fill['Volume'].abs().sum()
                fill_time.extend(list(fill['Time'].to_numpy() - insert['Time'].to_numpy()))

            if len(cancel) > 0:
                cancel_rate += cancel['Volume'].abs().sum()


        info[name]['Liquidity'] = dict(fill_rate= np.round(fill_rate / (fill_rate + cancel_rate), 2), fill_time_avg=np.round(np.mean(fill_time), 2))

def get_transition(competitors, info):
    # Keywords for each event
    keywords = [["Insert", "Ammend", "Cancel"], ["Tick"], ["Fill", "Hedge"]]
    # Measure how much of the stocks is filled and how fast it is filled
    for name, data in competitors.items():
        # transition probabilities (Quoting, Waiting, Spread)
        transitions = np.zeros((3, 3))
        operations = data['events']['Operation']
        for state, next_state in zip(operations, operations.iloc[1:]):
            if next_state == "Hedge" or state == next_state == "Insert": continue
            
            state_idx = list(i for i, item in enumerate(keywords) if state in item)
            next_state_idx = list(i for i, item in enumerate(keywords) if next_state in item)
            transitions[state_idx, next_state_idx] += 1

        transitions = transitions / np.sum(transitions, axis=0)
        
        info[name]['Transition'] = transitions
        
def markov_transitions(name, info):
    transition = info[name]["Liquidity"]["transitions"]
    G = nx.MultiDiGraph()
    edge_labels = {}
    states = ["Quoting", "Waiting", "Fill"]
    for i, state in enumerate(states):
        for j, next_state in enumerate(states):
            prob = transition[i][j]
            if prob <= 0:
                continue
            G.add_edge(state, next_state, weight=prob, label=f"{prob:.02f}")
            edge_labels[(state, next_state)] = f"{prob:.02f}"

    G.graph['edge'] = {'arrowsize': '0.6', 'splines': 'curved'}
    G.graph['graph'] = {'scale': '3'}

    A = to_agraph(G) 
    A.layout('dot')                                                                 
    A.draw(f"img/{name}_transition.png") 

def get_inventory_volatility(competitors, info):
    for name, data in competitors.items():
        position = data['events']['EtfPosition']
        info[name]['InventoryControl'] = dict(avg=np.round(np.mean(position), 2), std=np.round(np.std(position), 2))

info = defaultdict(lambda: dict())

get_profit_loss(competitors, info)
get_liquidity(competitors, info, 0.3)
get_inventory_volatility(competitors, info)

with open(output, "w") as fp:
    fp.write(json.dumps(info, sort_keys=True, indent=4, separators=(',', ':')))