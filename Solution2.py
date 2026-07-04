import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# note to self
# # second attempt - PCA for initial angle guess


y_offset = 42.0
omega = 0.3

theta_bounds = (np.radians(0.0), np.radians(50.0))
m_bounds = (-0.05, 0.05)
x_bounds = (0.0, 100.0)

def load_points(path):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]

def curve(t, theta, m, xshift):
    r = np.exp(m * np.abs(t)) * np.sin(omega * t)
    x = t * np.cos(theta) - r * np.sin(theta) + xshift
    y = y_offset + t * np.sin(theta) + r * np.cos(theta)
    return x, y

# rotate to canonical frame
def canonical(x, y, theta, xshift):
    dx, dy = x - xshift, y - y_offset
    c, s = np.cos(theta), np.sin(theta)
    u = dx * c + dy * s
    v = -dx * s + dy * c
    return u, v

def residuals(params, x, y):
    theta, m, xshift = params
    u, v = canonical(x, y, theta, xshift)
    return v - np.exp(m * u) * np.sin(omega * u)

def initial_guess(x, y):
    # PCA from linear algebra
    xc, yc = x - x.mean(), y - y.mean()
    cov = np.cov(np.vstack([xc, yc]))
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, np.argmax(eigvals)]
    theta = np.arctan2(major[1], major[0]) % np.pi
    if theta > np.pi / 2:
        theta -= np.pi
    xshift = x.mean() - (y.mean() - y_offset) / np.tan(theta)
    u, v = canonical(x, y, theta, xshift)
    sin_term = np.sin(omega * u)
    keep = np.abs(sin_term) > 0.3
    slope = np.polyfit(u[keep], np.log(np.abs(v[keep]) / np.abs(sin_term[keep])), 1)[0]
    m = np.clip(slope, *m_bounds)
    return np.array([theta, m, xshift])

def fit(x, y):
    p0 = initial_guess(x, y)
    print("initial guess: theta = %.4f deg, m = %.5f, xshift = %.4f" 
          % (np.degrees(p0[0]), p0[1], p0[2]))
    lower = [theta_bounds[0], m_bounds[0], x_bounds[0]]
    upper = [theta_bounds[1], m_bounds[1], x_bounds[1]]
    sol = least_squares(residuals, p0, args=(x, y), bounds=(lower, upper),
                        method="trf", xtol=1e-15, ftol=1e-15)
    return p0, sol

x, y = load_points("xy_data.csv")
print("fitting %d data points..." % len(x))
p0, sol = fit(x, y)
theta, m, xshift = sol.x
rms = np.sqrt(np.mean(sol.fun ** 2))
print("refined: theta = %.4f deg, m = %.5f, xshift = %.4f" 
      % (np.degrees(theta), m, xshift))
print("theta (rad) = %.6f" % theta)
print("rms residual = %.3e" % rms)

# plot
t = np.linspace(6, 60, 20000)
cx, cy = curve(t, theta, m, xshift)
plt.figure(figsize=(7, 5))
plt.scatter(x, y, s=5, alpha=0.4, label="data")
plt.plot(cx, cy, "r-", lw=1, label="recovered curve")
plt.gca().set_aspect("equal")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.savefig("fit.png", dpi=110)
print("saved fit.png")