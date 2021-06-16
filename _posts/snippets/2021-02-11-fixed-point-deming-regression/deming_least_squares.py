import os
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.figsize"] = (6, 6)


def deming_reg(x, y):
    a = np.sum(x * y)
    b = np.sum(np.square(x) - np.square(y))
    c = -a
    t1 = -b / (2 * a)
    t2 = np.sqrt(np.square(b) - 4 * a * c) / (2 * a)
    m_pos = t1 + t2
    m_neg = t1 - t2
    gen_x_star = lambda _m: (1 / (1 + _m ** 2)) * (_m * y + x)
    x_star_pos = gen_x_star(m_pos)
    s_pos = np.sum(np.square(x_star_pos - x) + np.square(m_pos * x_star_pos - y))
    x_star_neg = gen_x_star(m_neg)
    s_neg = np.sum(np.square(x_star_neg - x) + np.square(m_neg * x_star_neg - y))
    s = s_pos
    x_star = x_star_pos
    m = m_pos
    if s > s_neg:
        x_star = x_star_neg
        m = m_neg
    return x_star, m * x_star


save = True
x = np.linspace(-1, 4, 12)
y = x.copy()
np.random.seed(12)
y += 2 * (np.random.random(y.shape) - 0.5)
x_dem, y_dem = deming_reg(x, y)
plt.scatter(x, y, c="r")
plt.scatter(x_dem, y_dem, c="b")
plt.plot(x_dem, y_dem, c="b")
plt.plot((x, x_dem), (y, y_dem), c="k", ls="--")
plt.title("Deming Linear Regression")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim(-2, 5)
plt.ylim(-2, 5)
if save:
    rel_path = "assets/images/posts/2021-02-11-fixed-point-deming-regression/deming_least_squares.png"
    p = os.path.abspath(os.getcwd())
    for i in range(3):
        p = os.path.dirname(p)
    plt.savefig(os.path.relpath(os.path.join(p, rel_path)))
plt.show()
