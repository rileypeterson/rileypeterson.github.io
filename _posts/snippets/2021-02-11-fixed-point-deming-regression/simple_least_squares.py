import os
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.figsize"] = (6, 6)


save = True
x = np.linspace(-1, 4, 12)
y = x.copy()
np.random.seed(12)
y += 2 * (np.random.random(y.shape) - 0.5)
f = np.poly1d(np.polyfit(x, y, 1))
plt.scatter(x, y, c="r")
plt.scatter(x, f(x), c="b")
plt.plot(x, f(x), c="b")
plt.plot((x, x), (f(x), y), c="k", ls="--")
plt.title("Simple Linear Regression")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim(-2, 5)
plt.ylim(-2, 5)
if save:
    rel_path = "assets/images/posts/2021-02-11-fixed-point-deming-regression/simple_least_squares.png"
    p = os.path.abspath(os.getcwd())
    for i in range(3):
        p = os.path.dirname(p)
    plt.savefig(os.path.relpath(os.path.join(p, rel_path)))
plt.show()
