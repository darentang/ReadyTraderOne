import matplotlib.pyplot as plt
import numpy as np

def inventory_curves(self, q):
    if 0 <= q <= 20:
            return 20
    elif 20 < q <= 90 :
        return 20 + (q - 20) / 70 * 70
    elif q > 90:
        return 0
    elif -80 <= q < 0:
        return 10
    elif -90 <= q < -80:
        return 10 + (q + 80) 
    else:
        return 0


q = np.arange(-100, 100)
price_bid = [inventory_curves(None, x) for x in q]
price_sell = [inventory_curves(None, -x) for x in q]

plt.plot(q, price_bid)
plt.plot(q, price_sell)

plt.show()