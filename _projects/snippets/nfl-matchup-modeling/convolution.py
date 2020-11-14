import os
from scipy.stats import norm
from scipy.interpolate import interp1d
from scipy.signal import convolve
from scipy.integrate import quad
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["figure.figsize"] = (11, 8)
plt.style.use('fivethirtyeight')

offensive_scores = [20, 16, 21, 23, 30, 36, 20, 'BYE', 27]
offensive_scores = [i for i in offensive_scores if isinstance(i, int)]
defensive_scores = [38, 30, 20, 38, 23, 38, 34, 16, 'BYE']
defensive_scores = [i for i in defensive_scores if isinstance(i, int)]


team1 = "Houston Texans"
team2 = "Cleveland Browns"
xmin = 0
xmax = 55

def save_path():
    fname = os.path.abspath(__file__)
    fname = os.path.split(fname)[-1].replace(".py", ".png")
    rel_path = f"assets/images/nfl-matchup-modeling/{fname}"
    p = os.path.abspath(os.getcwd())
    for i in range(3):
        p = os.path.dirname(p)
    return os.path.relpath(os.path.join(p, rel_path))

def make_pdf(x, y):
    # This sucks, but this is the only way I can think to do this...
    f = interp1d(x, y, kind="cubic")
    tot = quad(f, xmin, xmax)
    mu = quad(lambda x1: x1 * f(x1), xmin, xmax)[0]
    std = np.sqrt(quad(lambda x1: (x1 - mu) ** 2 * f(x1), xmin, xmax))[0]
    return y / tot[0], mu, std

def return_pdf_params(scores):
    mean = np.mean(scores)
    std = np.std(scores)
    x = np.linspace(xmin, xmax, 1000)
    y = norm.pdf(x, mean, std)
    y, _, _ = make_pdf(x, y)
    return x, y, mean, std




def plot_matchup_distributions():
    x1, y1, mean1, std1 = return_pdf_params(offensive_scores)
    x2, y2, mean2, std2 = return_pdf_params(defensive_scores)
    plt.plot(x1, y1, label=f"{team1} Offense \n$\mu: {mean1 : .2f}$ ||| $\sigma : {std1 : .2f}$")
    plt.plot(x2, y2, label=f"{team2} Defense \n$\mu: {mean2 : .2f}$ ||| $\sigma : {std2 : .2f}$")

    values = convolve(y1, y2)
    x = np.linspace(xmin, xmax, len(values))
    values, mu, std = make_pdf(x, values)
    values, mu, std = make_pdf(x, values)
    plt.plot(x, values, label=f"Convolution: \n$\mu: {mu : .2f}$ ||| $\sigma : {std : .2f}$")

    plt.legend()
    plt.xlim(left=0, right=55)
    plt.xlabel("Points")
    plt.ylabel("Probability")
    plt.title(f"Offensive Points: {team1} | Defensive Points: {team2}")
    plt.savefig(save_path())


plot_matchup_distributions()
