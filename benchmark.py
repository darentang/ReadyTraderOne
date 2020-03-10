"""
Evaluation of models
"""
import pandas as pd
import numpy as np
import networkx as nx
from networkx.drawing.nx_agraph import to_agraph 

import matplotlib.pyplot as plt
from collections import defaultdict

filename = "ready_trader_one/match_events.csv"

df = pd.read_csv(filename)
bots = df["Competitor"]

# stores dataframe for each competitor
competitors = dict()

for bot in bots.unique():
    competitors[bot] = dict(events=df.query(f"Competitor == '{bot}'"))
     

def get_profit_loss(competitors, info):
    for name, data in competitors.items():
        profit_loss = data['events']['ProfitLoss']
        info[name]['PnL'] = dict(last=profit_loss.iloc[-1], avg=np.mean(profit_loss), std=np.std(profit_loss))

def get_liquidity(competitors, info):
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
        
        info[name]['Liquidity'] = dict(transitions=transitions)
        
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
        position = data['events']['Position']
        info[name]['Volatility'] = dict(std=np.std(position))

info = defaultdict(lambda: dict())

get_profit_loss(competitors, info)
get_liquidity(competitors, info)


print(info)