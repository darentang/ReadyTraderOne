import pandas as pd

filename = "ready_trader_one/match_events.csv"
teams = ["TeaMaster"]

df_whole = pd.read_csv(filename)
for team in teams:
    df = df_whole.query(f"Competitor == '{team}'")

    fills = df.query("Operation == 'Fill'")
    hedges = df.query("Operation == 'Hedge'")

    for (_, fill), (_,hedge) in zip(fills.iterrows(), hedges.iterrows()):
        fill_total = - (fill["Price"] * fill["Volume"])
        hedge_total = (hedge["Price"] * hedge["Volume"])

        if fill["Side"] == "B":
            profit = hedge_total - fill_total
        elif fill["Side"] == "S":
            profit = fill_total - hedge_total

        fee = fill["Fee"]

        if profit - fee < 0:
            print("\n       --Net Loss--        $%.2f" % (profit-fee))
            print(pd.concat([fill, hedge], axis=1))

