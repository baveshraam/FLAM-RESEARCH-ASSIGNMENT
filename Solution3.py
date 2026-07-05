import numpy as np

import matplotlib.pyplot as plt

from scipy.optimize import least_squares



# ok so the problem is find theta, M, X in this curve

# x = t*cos(theta) - e^(M*|t|)*sin(0.3t)*sin(theta) + X

# y = 42 + t*sin(theta) + e^(M*|t|)*sin(0.3t)*cos(theta)







y_off = 42.0

w = 0.3

theta_min = np.radians(0.0)

theta_max = np.radians(50.0)

m_min = -0.05

m_max = 0.05

x_min = 0.0

x_max = 100.0



def load(path):

  d = np.loadtxt(path, delimiter=",", skiprows=1)

  return d[:, 0], d[:, 1]





# so I'm going to calculate the curve for given params

def calc_curve(t, th, mm, xs):

  # r = e^(M*|t|) * sin(0.3*t)

  r = np.exp(mm * np.abs(t)) * np.sin(w * t)

  xx = t * np.cos(th) - r * np.sin(th) + xs

  yy = y_off + t * np.sin(th) + r * np.cos(th)

  return xx, yy





# rotate coords so curve aligns with axes



def rotate(x, y, th, xs):

  dx = x - xs

  dy = y - y_off

  c = np.cos(th)

  s = np.sin(th)

  # u is along the curve

  u = dx * c + dy * s

  # v is perpendicular

  v = -dx * s + dy * c

  return u, v





# residuals for least squares

# in rotated frame, v should equal e^(M*u)*sin(0.3*u)

def resid(p, x, y):

  th, mm, xs = p

  u, v = rotate(x, y, th, xs)

  return v - np.exp(mm * u) * np.sin(w * u)





# initial guess using PCA



def guess(x, y):

  # center data

  xc = x - np.mean(x)

  yc = y - np.mean(y)

   

  # covariance matrix

  cov = np.cov(np.vstack([xc, yc]))

   

  # eigenvalues and eigenvectors

  vals, vecs = np.linalg.eigh(cov)

   

  # major axis direction

  major = vecs[:, np.argmax(vals)]

   

  # angle of major axis

  th = np.arctan2(major[1], major[0]) % np.pi

  if th > np.pi / 2:

    th -= np.pi

   

  

  xs = np.mean(x) - (np.mean(y) - y_off) / np.tan(th)



  u, v = rotate(x, y, th, xs)

  sin_t = np.sin(w * u)

   

  # only use points where sin term is not near zero

  # to avoid division by zero

  mask = np.abs(sin_t) > 0.3

   

  # log(|v|/|sin|) vs u should be linear with slope M

  yy = np.log(np.abs(v[mask]) / np.abs(sin_t[mask]))

  xx = u[mask]

   

  # fit line and  clip to bounds

  slope = np.polyfit(xx, yy, 1)[0]

   

 

  mm = np.clip(slope, m_min, m_max)

   

  print("guess: theta=%.4f deg, M=%.5f, X=%.4f" % (np.degrees(th), mm, xs))

   

  return np.array([th, mm, xs])





# main fitting function

def fit(x, y):

  p0 = guess(x, y)

   

  # bounds for least_squares

  low = [theta_min, m_min, x_min]

  high = [theta_max, m_max, x_max]

   

  #  optimization for the win

  sol = least_squares(resid, p0, args=(x, y), bounds=(low, high),

            method="trf", xtol=1e-15, ftol=1e-15)

   

  return p0, sol



if __name__ == "__main__":

  print("loading data...")

  x, y = load("xy_data.csv")

  print("loaded %d points" % len(x))

   

  print("\nfitting...")

  p0, sol = fit(x, y)

   

  th, mm, xs = sol.x

  rms = np.sqrt(np.mean(sol.fun ** 2))

   

  print("\n RESULTS :")

  print("theta = %.6f deg (%.6f rad)" % (np.degrees(th), th))

  print("M = %.6f" % mm)

  print("X = %.6f" % xs)

  print("RMS residual = %.3e" % rms)

   

  # so if i compute distance from data to curve we can verify 

  print("\nverifyyyyyyyyyyyyyy")

  t = np.linspace(6, 60, 20000)

  cx, cy = calc_curve(t, th, mm, xs)

   

  # so this is to findd distance from each data point to nearest curve point 

  dists = np.min(np.hypot(x[:, None] - cx, y[:, None] - cy), axis=1)

  print("max distance = %.3e" % np.max(dists))

  print("mean distance = %.3e" % np.mean(dists))

   

  # plotsssssssssssssssssssss

  print("\nsaving plot...")

  plt.figure(figsize=(7, 5))

  plt.scatter(x, y, s=5, alpha=0.4, label="data")

  plt.plot(cx, cy, "r-", lw=1, label="fit")

  plt.xlabel("x")

  plt.ylabel("y")

  plt.legend()

  plt.tight_layout()

  plt.savefig("fit.png", dpi=110)

  print("saved fit.png")
  
