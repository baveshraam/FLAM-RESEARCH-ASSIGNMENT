# attempt 1

import numpy as np
import matplotlib.pyplot as plt



y_offset = 42.0
omega = 0.3

def load_points(path):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]

def curve(t, theta, m, xshift):
    r = np.exp(m * np.abs(t)) * np.sin(omega * t)
    x = t * np.cos(theta) - r * np.sin(theta) + xshift
    y = y_offset + t * np.sin(theta) + r * np.cos(theta)
    return x, y

x, y = load_points("xy_data.csv")

# note to self, grid search - way too slow
# theta_range = np.linspace(0, np.radians(50), 50)
# m_range = np.linspace(-0.05, 0.05, 20)
# xshift_range = np.linspace(0, 100, 50)
# this would be 50*20*50 = 50000 iterations, too slow

# just plot the data for now
plt.figure(figsize=(7, 5))
plt.scatter(x, y, s=5, alpha=0.4)
plt.xlabel("x")
plt.ylabel("y")
plt.title("raw data - looks rotated")
plt.savefig("data_raw.png", dpi=110)
print("plotted raw data, need better approach")