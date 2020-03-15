import numpy as np
import matplotlib.pyplot as plt

class Orderbook:
    def __init__(self, name):
        self.bid_volumes = None
        self.ask_volumes = None
        self.bid_prices = None
        self.ask_prices = None
        self.name = name
        self.figure = plt.figure()
        self.max_length = 10
        self.history = np.zeros((self.max_length, 2))
        self.gradient_length = 10
        self.fit_degree = 5
        self.fit_coeff = np.zeros(self.fit_degree + 1)
        self.i = 0

    def get_table(self):
        return np.vstack((self.bid_volumes, self.bid_prices, self.ask_prices, self.ask_volumes)).astype(str).T

    def update_orders(self, bid_volumes, bid_prices, ask_prices, ask_volumes, time=0):
        self.bid_volumes = bid_volumes
        self.ask_volumes = ask_volumes
        self.bid_prices = bid_prices
        self.ask_prices = ask_prices
        self.total_volume = np.sum([bid_volumes, ask_volumes])
        
        self.update(self.midpoint(), time)
        

    def update(self, price, time):
        if self.i < self.max_length:
            self.history[self.i, :] = [time, price]
        else:
            self.history = np.roll(self.history, -1, axis=0)
            self.history[-1, :] = [time, price]
        self.fit()
        self.i += 1

    def gradient(self, time):
        return np.polyval(np.polyder(self.fit_coeff, 1), time)

    def acceleration(self, time):
        return np.polyval(np.polyder(self.fit_coeff, 2), time)

    def predict(self, time):
        return np.polyval(self.fit_coeff, time)

    def fit(self):
        if self.i < self.fit_degree:
            return

        if self.i > self.max_length:
            i = self.max_length - 1
        else:
            i = self.i
        
        if self.i > self.gradient_length:
            l = self.gradient_length - 1
        else:
            l = i
        
        
        X = self.history[i  - l:i + 1, 0]
        Y = self.history[i  - l:i + 1, 1]
        self.fit_coeff = np.polyfit(X, Y, deg=self.fit_degree)

    def best_bid(self):
        return self.bid_prices[0]

    def best_ask(self):
        return self.ask_prices[0]

    def midpoint(self):
        if self.total_volume != 0:
            return (np.dot(self.bid_prices, self.bid_volumes) + np.dot(self.ask_prices, self.ask_volumes)) / self.total_volume
        else:
            return 0


def gauss(x, mu, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp( -0.5  * ( (x - mu) / sigma) ** 2)

sigma = 0.1
sigma_v = 2
prob = [0.6, 0.4]
N = 12
ask_prices = np.zeros((N , 5))
ask_volumes = np.zeros((N , 5))
bid_prices = np.zeros((N , 5))
bid_volumes = np.zeros((N , 5))
midpoints = np.zeros(N)
predicted_midpoints = np.zeros(N)
predicted_gradient = np.zeros(N)
time = np.arange(N)

midpoint = 0
probs = np.random.rand(N)

orderbook = Orderbook("Test")
for i in range(N):
    if probs[i] < prob[0]:
        midpoint += np.abs(np.random.normal() * sigma)
    else:
        midpoint -= np.abs(np.random.normal() * sigma)
    midpoints[i] = midpoint

    ask_prices[i, :] = midpoint + np.abs(np.random.normal(size=5)) * sigma
    bid_prices[i, :] = midpoint - np.abs(np.random.normal(size=5)) * sigma

    ask_volumes[i, :] = np.round(100 * gauss(ask_prices[i, :], midpoint, sigma_v))
    bid_volumes[i, :] = np.round(100 * gauss(bid_prices[i, :], midpoint, sigma_v))

    orderbook.update_orders(bid_volumes[i, :], bid_prices[i, :], ask_prices[i, :], ask_volumes[i, :], time=time[i])
    predicted_midpoints[i] = orderbook.midpoint()

    plt.scatter(time[i] * np.ones(5), ask_prices[i, :], c="r", s=ask_volumes[i, :])
    plt.scatter(time[i] * np.ones(5), bid_prices[i, :], c="b", s=bid_volumes[i, :])

plt.figure(1)

plt.plot(time, midpoints)
plt.plot(time, predicted_midpoints)
plt.plot(time[-10:], orderbook.predict(time[-10:]))
plt.plot(time[-1] + np.array([0, 1]), predicted_midpoints[-1] + np.array([0, orderbook.gradient(time[-1])]))
plt.show()


