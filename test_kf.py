import matplotlib.pyplot as plt
import numpy as np

class KalmanFilter:
    def __init__(self, R, Q, X0, P0):
        self.R = R
        self.Q = Q
        self.X = X0
        self.P = P0
        self.H = np.array([1, 0])
        self.K = np.zeros(2)
        self.t = 0

    def gamma(self, t):
        return np.array([0.5 * (t - self.t) ** 2, (t - self.t)])
        
    def A(self, t):
        return np.array([[1, t - self.t], [0, 1]])
    
    def predict(self, t):
        # state vector 
        self.X = self.A(t) @ self.X
        self.P = self.A(t) @ self.P @ self.A(t).T + self.Q * self.gamma(t) @ self.gamma(t).T
        # self.P = self.A(t) @ self.P @ self.A(t).T + self.Q 

    def update(self, z, t):
        self.K = self.P @ self.H.T  / (self.H @ self.P @ self.H.T + self.R)
        self.X = self.X + self.K * (z - self.H @ self.X)
        self.P = (np.identity(2) - self.K @ self.H) @ self.P
        self.t = t

    def cov(self, sigma1, sigma2):
        cov_matrix = np.outer([sigma1, sigma2], [sigma1, sigma2])
        return np.diag(np.diag(cov_matrix))


T = 100
N = T
noise = 10

dt = T / (N - 1)

t = np.linspace(0, T, N)
x = (2 *t ** 2 + t) * 0.01
# observations
z = np.random.normal(x, noise, size=N)
# velocities
v = (z[:-1] - z[1:]) / dt

x_hat = np.zeros(N)

# Initial guess
X0 = np.array([np.mean(z), np.mean(v)])
# P0 = np.cov(z[1:], v)
P0 = np.identity(2)

plt.plot(t, x)
plt.scatter(t, z, s=2, c='r')

kf = KalmanFilter(10, 10, X0, P0)

for i, (observation, time) in enumerate(zip(z, t)):
    kf.predict(time)
    x_hat[i] = kf.X[0]
    kf.update(observation, time)

plt.plot(t, x_hat)

plt.show()