import os
from scipy.stats import norm
from scipy.interpolate import interp1d
from scipy.integrate import quad
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["figure.figsize"] = (11, 8)
plt.style.use('fivethirtyeight')

scores = [20, 16, 21, 23, 30, 36, 20, 'BYE', 27]
scores = [i for i in scores if isinstance(i, int)]

team = "Houston Texans"
xmin = 0
xmax = 55



def make_pdf(x, y):
    # This sucks, but this is the only way I can think to do this...
    f = interp1d(x, y, kind="cubic")
    tot = quad(f, xmin, xmax)
    mu = quad(lambda x1: x1 * f(x1), xmin, xmax)[0]
    std = np.sqrt(quad(lambda x1: (x1 - mu) ** 2 * f(x1), xmin, xmax))[0]
    return y / tot[0], mu, std



def plot_offensive_points(scores):
    mean = np.mean(scores)
    std = np.std(scores)
    x = np.linspace(xmin, xmax, 1000)
    y = norm.pdf(x, mean, std)
    y, _, _ = make_pdf(x, y)
    plt.plot(x, y, label=f"$\mu: {mean : .2f}$ ||| $\sigma : {std : .2f}$")
    w = 0.2
    mask = (mean - std < x) & (mean + std > x)

    plt.fill(x[mask], y[mask], alpha=0.25, color="blue")
    plt.fill_between([mean - std, mean + std], min(y[mask]), color="blue", alpha=0.25)
    plt.fill_between([mean - w, mean + w], max(y[mask]), color="red", zorder=100)
    plt.legend()
    plt.xlim(left=0, right=55)
    plt.xlabel("Points")
    plt.ylabel("Probability")
    plt.title(f"Offensive Points: {team}")
    rel_path = "assets/images/nfl-matchup-modeling/gaussian1.png"
    p = os.path.abspath(os.getcwd())
    for i in range(3):
        p = os.path.dirname(p)
    plt.savefig(os.path.relpath(os.path.join(p, rel_path)))


plot_offensive_points(scores)