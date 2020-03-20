import pandas as pd

filename = "ready_trader_one/match_events.csv"
teams = ["TeaMaster"]

df_whole = pd.read_csv(filename)
for team in teams:
    df = df_whole.query(f"Competitor == '{team}'")

    fills = df.query("Operation == 'Fill'")
    hedges = df.query("Operation == 'Hedge'")

    for (_, fill), (_,hedge) in zip(fills.iterrows(), hedges.iterrows()):
    


        fill_total = (fill["Price"] * fill["Volume"])
        hedge_total = (hedge["Price"] * hedge["Volume"])

        print(fill_total + hedge_total)

