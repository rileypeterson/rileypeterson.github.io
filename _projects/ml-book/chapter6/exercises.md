---
layout: exercises
chapter: 6
chapter-title: Decision Trees
permalink: /ml-book/chapter6/exercises.html
---

## Exercise 1

The approximate depth of a Decision Tree trained (without restrictions) on a training set with 1 million instances is $log(10^6) \approx 20$.

## Exercise 2

A node's Gini impurity is generally lower than it's parents. This is because the gini impurity is weighted by the size of the node.

## Exercise 3

If a Decision Tree is overfitting the training set, it is a good idea to decrease `max_depth`. You would be reducing the complexity of the model which combats overfitting.

Hopefully, this is visible from above. In the second figure, the y value is insignificant because it is vastly outscaled by x. Clearly, when they are of equal scales both x and y are significant in determining a boundary. 

## Exercise 4

If a Decision Tree is underfitting it is neither a good idea or bad idea to scale the input features. Scaling the input features won't have any effect.

## Exercise 5

If it takes an hour to train 1 million instances, on 10 million instances we'd expect the following:

1 hour / x hours = (n * m * log(m)) / (n * 10m * log(10m))

Re-arranging and simplifying:

x = (10 * log(10m)) /  (log(m))

**x ~ 11.66 hours**

## Exercise 6

No, presort is only beneficial for training sets with < 1,000 samples. I don't even see `presort=True` in the docs anymore.

## Exercise 7

Train and fine-tune a Decision Tree for the moons dataset.

### Exercise 7a

Generate a moons dataset using `make_moons(n_samples=10000, noise=0.4)`


```python
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=10_000, noise=0.4, random_state=42)
```

### Exercise 7b

Split it into a training set and a test set using `train_test_split()`


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### Exercise 7c

Use grid search with cross-validation (with the help of the `GridSearchCV` class) to find good hyperparameter values for a `DecisionTreeClassifier`. Hint: try various values for `max_leaf_nodes`.


```python
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

tree_clf = DecisionTreeClassifier(random_state=42)

params = {"max_leaf_nodes": list(range(2, 21)), "min_samples_split": list(range(2, 10))}

gs = GridSearchCV(tree_clf, params, cv=5, scoring="accuracy", verbose=3)
gs.fit(X_train, y_train)
```

    Fitting 5 folds for each of 152 candidates, totalling 760 fits
    [CV 1/5] END max_leaf_nodes=2, min_samples_split=2;, score=0.751 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=2, min_samples_split=2;, score=0.762 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=2, min_samples_split=2;, score=0.787 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=2, min_samples_split=2;, score=0.782 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=2, min_samples_split=2;, score=0.767 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=2, min_samples_split=3;, score=0.751 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=2, min_samples_split=3;, score=0.762 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=2, min_samples_split=3;, score=0.787 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=2, min_samples_split=3;, score=0.782 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=2, min_samples_split=3;, score=0.767 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=2, min_samples_split=4;, score=0.751 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=2, min_samples_split=4;, score=0.762 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=2, min_samples_split=4;, score=0.787 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=2, min_samples_split=4;, score=0.782 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=2, min_samples_split=4;, score=0.767 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=2, min_samples_split=5;, score=0.751 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=2, min_samples_split=5;, score=0.762 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=2, min_samples_split=5;, score=0.787 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=2, min_samples_split=5;, score=0.782 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=2, min_samples_split=5;, score=0.767 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=2, min_samples_split=6;, score=0.751 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=2, min_samples_split=6;, score=0.762 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=2, min_samples_split=6;, score=0.787 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=2, min_samples_split=6;, score=0.782 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=2, min_samples_split=6;, score=0.767 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=2, min_samples_split=7;, score=0.751 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=2, min_samples_split=7;, score=0.762 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=2, min_samples_split=7;, score=0.787 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=2, min_samples_split=7;, score=0.782 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=2, min_samples_split=7;, score=0.767 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=2, min_samples_split=8;, score=0.751 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=2, min_samples_split=8;, score=0.762 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=2, min_samples_split=8;, score=0.787 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=2, min_samples_split=8;, score=0.782 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=2, min_samples_split=8;, score=0.767 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=2, min_samples_split=9;, score=0.751 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=2, min_samples_split=9;, score=0.762 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=2, min_samples_split=9;, score=0.787 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=2, min_samples_split=9;, score=0.782 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=2, min_samples_split=9;, score=0.767 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=3, min_samples_split=2;, score=0.797 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=3, min_samples_split=2;, score=0.823 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=3, min_samples_split=2;, score=0.828 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=3, min_samples_split=2;, score=0.828 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=3, min_samples_split=2;, score=0.809 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=3, min_samples_split=3;, score=0.797 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=3, min_samples_split=3;, score=0.823 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=3, min_samples_split=3;, score=0.828 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=3, min_samples_split=3;, score=0.828 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=3, min_samples_split=3;, score=0.809 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=3, min_samples_split=4;, score=0.797 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=3, min_samples_split=4;, score=0.823 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=3, min_samples_split=4;, score=0.828 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=3, min_samples_split=4;, score=0.828 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=3, min_samples_split=4;, score=0.809 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=3, min_samples_split=5;, score=0.797 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=3, min_samples_split=5;, score=0.823 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=3, min_samples_split=5;, score=0.828 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=3, min_samples_split=5;, score=0.828 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=3, min_samples_split=5;, score=0.809 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=3, min_samples_split=6;, score=0.797 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=3, min_samples_split=6;, score=0.823 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=3, min_samples_split=6;, score=0.828 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=3, min_samples_split=6;, score=0.828 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=3, min_samples_split=6;, score=0.809 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=3, min_samples_split=7;, score=0.797 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=3, min_samples_split=7;, score=0.823 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=3, min_samples_split=7;, score=0.828 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=3, min_samples_split=7;, score=0.828 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=3, min_samples_split=7;, score=0.809 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=3, min_samples_split=8;, score=0.797 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=3, min_samples_split=8;, score=0.823 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=3, min_samples_split=8;, score=0.828 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=3, min_samples_split=8;, score=0.828 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=3, min_samples_split=8;, score=0.809 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=3, min_samples_split=9;, score=0.797 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=3, min_samples_split=9;, score=0.823 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=3, min_samples_split=9;, score=0.828 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=3, min_samples_split=9;, score=0.828 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=3, min_samples_split=9;, score=0.809 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=4, min_samples_split=2;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=4, min_samples_split=2;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=4, min_samples_split=2;, score=0.861 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=4, min_samples_split=2;, score=0.857 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=4, min_samples_split=2;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=4, min_samples_split=3;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=4, min_samples_split=3;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=4, min_samples_split=3;, score=0.861 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=4, min_samples_split=3;, score=0.857 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=4, min_samples_split=3;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=4, min_samples_split=4;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=4, min_samples_split=4;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=4, min_samples_split=4;, score=0.861 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=4, min_samples_split=4;, score=0.857 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=4, min_samples_split=4;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=4, min_samples_split=5;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=4, min_samples_split=5;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=4, min_samples_split=5;, score=0.861 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=4, min_samples_split=5;, score=0.857 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=4, min_samples_split=5;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=4, min_samples_split=6;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=4, min_samples_split=6;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=4, min_samples_split=6;, score=0.861 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=4, min_samples_split=6;, score=0.857 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=4, min_samples_split=6;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=4, min_samples_split=7;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=4, min_samples_split=7;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=4, min_samples_split=7;, score=0.861 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=4, min_samples_split=7;, score=0.857 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=4, min_samples_split=7;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=4, min_samples_split=8;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=4, min_samples_split=8;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=4, min_samples_split=8;, score=0.861 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=4, min_samples_split=8;, score=0.857 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=4, min_samples_split=8;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=4, min_samples_split=9;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=4, min_samples_split=9;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=4, min_samples_split=9;, score=0.861 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=4, min_samples_split=9;, score=0.857 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=4, min_samples_split=9;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=5, min_samples_split=2;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=5, min_samples_split=2;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=5, min_samples_split=2;, score=0.861 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=5, min_samples_split=2;, score=0.857 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=5, min_samples_split=2;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=5, min_samples_split=3;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=5, min_samples_split=3;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=5, min_samples_split=3;, score=0.861 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=5, min_samples_split=3;, score=0.857 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=5, min_samples_split=3;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=5, min_samples_split=4;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=5, min_samples_split=4;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=5, min_samples_split=4;, score=0.861 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=5, min_samples_split=4;, score=0.857 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=5, min_samples_split=4;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=5, min_samples_split=5;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=5, min_samples_split=5;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=5, min_samples_split=5;, score=0.861 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=5, min_samples_split=5;, score=0.857 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=5, min_samples_split=5;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=5, min_samples_split=6;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=5, min_samples_split=6;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=5, min_samples_split=6;, score=0.861 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=5, min_samples_split=6;, score=0.857 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=5, min_samples_split=6;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=5, min_samples_split=7;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=5, min_samples_split=7;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=5, min_samples_split=7;, score=0.861 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=5, min_samples_split=7;, score=0.857 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=5, min_samples_split=7;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=5, min_samples_split=8;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=5, min_samples_split=8;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=5, min_samples_split=8;, score=0.861 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=5, min_samples_split=8;, score=0.857 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=5, min_samples_split=8;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=5, min_samples_split=9;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=5, min_samples_split=9;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=5, min_samples_split=9;, score=0.861 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=5, min_samples_split=9;, score=0.857 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=5, min_samples_split=9;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=6, min_samples_split=2;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=6, min_samples_split=2;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=6, min_samples_split=2;, score=0.861 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=6, min_samples_split=2;, score=0.857 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=6, min_samples_split=2;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=6, min_samples_split=3;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=6, min_samples_split=3;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=6, min_samples_split=3;, score=0.861 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=6, min_samples_split=3;, score=0.857 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=6, min_samples_split=3;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=6, min_samples_split=4;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=6, min_samples_split=4;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=6, min_samples_split=4;, score=0.861 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=6, min_samples_split=4;, score=0.857 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=6, min_samples_split=4;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=6, min_samples_split=5;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=6, min_samples_split=5;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=6, min_samples_split=5;, score=0.861 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=6, min_samples_split=5;, score=0.857 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=6, min_samples_split=5;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=6, min_samples_split=6;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=6, min_samples_split=6;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=6, min_samples_split=6;, score=0.861 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=6, min_samples_split=6;, score=0.857 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=6, min_samples_split=6;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=6, min_samples_split=7;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=6, min_samples_split=7;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=6, min_samples_split=7;, score=0.861 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=6, min_samples_split=7;, score=0.857 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=6, min_samples_split=7;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=6, min_samples_split=8;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=6, min_samples_split=8;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=6, min_samples_split=8;, score=0.861 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=6, min_samples_split=8;, score=0.857 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=6, min_samples_split=8;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=6, min_samples_split=9;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=6, min_samples_split=9;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=6, min_samples_split=9;, score=0.861 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=6, min_samples_split=9;, score=0.857 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=6, min_samples_split=9;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=7, min_samples_split=2;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=7, min_samples_split=2;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=7, min_samples_split=2;, score=0.861 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=7, min_samples_split=2;, score=0.857 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=7, min_samples_split=2;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=7, min_samples_split=3;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=7, min_samples_split=3;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=7, min_samples_split=3;, score=0.861 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=7, min_samples_split=3;, score=0.857 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=7, min_samples_split=3;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=7, min_samples_split=4;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=7, min_samples_split=4;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=7, min_samples_split=4;, score=0.861 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=7, min_samples_split=4;, score=0.857 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=7, min_samples_split=4;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=7, min_samples_split=5;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=7, min_samples_split=5;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=7, min_samples_split=5;, score=0.861 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=7, min_samples_split=5;, score=0.857 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=7, min_samples_split=5;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=7, min_samples_split=6;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=7, min_samples_split=6;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=7, min_samples_split=6;, score=0.861 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=7, min_samples_split=6;, score=0.857 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=7, min_samples_split=6;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=7, min_samples_split=7;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=7, min_samples_split=7;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=7, min_samples_split=7;, score=0.861 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=7, min_samples_split=7;, score=0.857 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=7, min_samples_split=7;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=7, min_samples_split=8;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=7, min_samples_split=8;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=7, min_samples_split=8;, score=0.861 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=7, min_samples_split=8;, score=0.857 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=7, min_samples_split=8;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=7, min_samples_split=9;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=7, min_samples_split=9;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=7, min_samples_split=9;, score=0.861 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=7, min_samples_split=9;, score=0.857 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=7, min_samples_split=9;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=8, min_samples_split=2;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=8, min_samples_split=2;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=8, min_samples_split=2;, score=0.861 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=8, min_samples_split=2;, score=0.857 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=8, min_samples_split=2;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=8, min_samples_split=3;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=8, min_samples_split=3;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=8, min_samples_split=3;, score=0.861 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=8, min_samples_split=3;, score=0.857 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=8, min_samples_split=3;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=8, min_samples_split=4;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=8, min_samples_split=4;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=8, min_samples_split=4;, score=0.861 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=8, min_samples_split=4;, score=0.857 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=8, min_samples_split=4;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=8, min_samples_split=5;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=8, min_samples_split=5;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=8, min_samples_split=5;, score=0.861 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=8, min_samples_split=5;, score=0.857 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=8, min_samples_split=5;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=8, min_samples_split=6;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=8, min_samples_split=6;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=8, min_samples_split=6;, score=0.861 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=8, min_samples_split=6;, score=0.857 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=8, min_samples_split=6;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=8, min_samples_split=7;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=8, min_samples_split=7;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=8, min_samples_split=7;, score=0.861 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=8, min_samples_split=7;, score=0.857 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=8, min_samples_split=7;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=8, min_samples_split=8;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=8, min_samples_split=8;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=8, min_samples_split=8;, score=0.861 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=8, min_samples_split=8;, score=0.857 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=8, min_samples_split=8;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=8, min_samples_split=9;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=8, min_samples_split=9;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=8, min_samples_split=9;, score=0.861 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=8, min_samples_split=9;, score=0.857 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=8, min_samples_split=9;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=9, min_samples_split=2;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=9, min_samples_split=2;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=9, min_samples_split=2;, score=0.849 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=9, min_samples_split=2;, score=0.856 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=9, min_samples_split=2;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=9, min_samples_split=3;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=9, min_samples_split=3;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=9, min_samples_split=3;, score=0.849 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=9, min_samples_split=3;, score=0.856 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=9, min_samples_split=3;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=9, min_samples_split=4;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=9, min_samples_split=4;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=9, min_samples_split=4;, score=0.849 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=9, min_samples_split=4;, score=0.856 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=9, min_samples_split=4;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=9, min_samples_split=5;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=9, min_samples_split=5;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=9, min_samples_split=5;, score=0.849 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=9, min_samples_split=5;, score=0.856 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=9, min_samples_split=5;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=9, min_samples_split=6;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=9, min_samples_split=6;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=9, min_samples_split=6;, score=0.849 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=9, min_samples_split=6;, score=0.856 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=9, min_samples_split=6;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=9, min_samples_split=7;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=9, min_samples_split=7;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=9, min_samples_split=7;, score=0.849 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=9, min_samples_split=7;, score=0.856 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=9, min_samples_split=7;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=9, min_samples_split=8;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=9, min_samples_split=8;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=9, min_samples_split=8;, score=0.849 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=9, min_samples_split=8;, score=0.856 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=9, min_samples_split=8;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=9, min_samples_split=9;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=9, min_samples_split=9;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=9, min_samples_split=9;, score=0.849 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=9, min_samples_split=9;, score=0.856 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=9, min_samples_split=9;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=10, min_samples_split=2;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=10, min_samples_split=2;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=10, min_samples_split=2;, score=0.849 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=10, min_samples_split=2;, score=0.856 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=10, min_samples_split=2;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=10, min_samples_split=3;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=10, min_samples_split=3;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=10, min_samples_split=3;, score=0.849 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=10, min_samples_split=3;, score=0.856 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=10, min_samples_split=3;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=10, min_samples_split=4;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=10, min_samples_split=4;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=10, min_samples_split=4;, score=0.849 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=10, min_samples_split=4;, score=0.856 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=10, min_samples_split=4;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=10, min_samples_split=5;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=10, min_samples_split=5;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=10, min_samples_split=5;, score=0.849 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=10, min_samples_split=5;, score=0.856 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=10, min_samples_split=5;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=10, min_samples_split=6;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=10, min_samples_split=6;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=10, min_samples_split=6;, score=0.849 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=10, min_samples_split=6;, score=0.856 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=10, min_samples_split=6;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=10, min_samples_split=7;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=10, min_samples_split=7;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=10, min_samples_split=7;, score=0.849 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=10, min_samples_split=7;, score=0.856 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=10, min_samples_split=7;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=10, min_samples_split=8;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=10, min_samples_split=8;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=10, min_samples_split=8;, score=0.849 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=10, min_samples_split=8;, score=0.856 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=10, min_samples_split=8;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=10, min_samples_split=9;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=10, min_samples_split=9;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=10, min_samples_split=9;, score=0.849 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=10, min_samples_split=9;, score=0.856 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=10, min_samples_split=9;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=11, min_samples_split=2;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=11, min_samples_split=2;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=11, min_samples_split=2;, score=0.849 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=11, min_samples_split=2;, score=0.856 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=11, min_samples_split=2;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=11, min_samples_split=3;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=11, min_samples_split=3;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=11, min_samples_split=3;, score=0.849 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=11, min_samples_split=3;, score=0.856 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=11, min_samples_split=3;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=11, min_samples_split=4;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=11, min_samples_split=4;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=11, min_samples_split=4;, score=0.849 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=11, min_samples_split=4;, score=0.856 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=11, min_samples_split=4;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=11, min_samples_split=5;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=11, min_samples_split=5;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=11, min_samples_split=5;, score=0.849 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=11, min_samples_split=5;, score=0.856 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=11, min_samples_split=5;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=11, min_samples_split=6;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=11, min_samples_split=6;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=11, min_samples_split=6;, score=0.849 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=11, min_samples_split=6;, score=0.856 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=11, min_samples_split=6;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=11, min_samples_split=7;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=11, min_samples_split=7;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=11, min_samples_split=7;, score=0.849 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=11, min_samples_split=7;, score=0.856 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=11, min_samples_split=7;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=11, min_samples_split=8;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=11, min_samples_split=8;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=11, min_samples_split=8;, score=0.849 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=11, min_samples_split=8;, score=0.856 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=11, min_samples_split=8;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=11, min_samples_split=9;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=11, min_samples_split=9;, score=0.863 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=11, min_samples_split=9;, score=0.849 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=11, min_samples_split=9;, score=0.856 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=11, min_samples_split=9;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=12, min_samples_split=2;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=12, min_samples_split=2;, score=0.871 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=12, min_samples_split=2;, score=0.858 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=12, min_samples_split=2;, score=0.854 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=12, min_samples_split=2;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=12, min_samples_split=3;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=12, min_samples_split=3;, score=0.871 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=12, min_samples_split=3;, score=0.858 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=12, min_samples_split=3;, score=0.854 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=12, min_samples_split=3;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=12, min_samples_split=4;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=12, min_samples_split=4;, score=0.871 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=12, min_samples_split=4;, score=0.858 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=12, min_samples_split=4;, score=0.854 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=12, min_samples_split=4;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=12, min_samples_split=5;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=12, min_samples_split=5;, score=0.871 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=12, min_samples_split=5;, score=0.858 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=12, min_samples_split=5;, score=0.854 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=12, min_samples_split=5;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=12, min_samples_split=6;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=12, min_samples_split=6;, score=0.871 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=12, min_samples_split=6;, score=0.858 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=12, min_samples_split=6;, score=0.854 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=12, min_samples_split=6;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=12, min_samples_split=7;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=12, min_samples_split=7;, score=0.871 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=12, min_samples_split=7;, score=0.858 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=12, min_samples_split=7;, score=0.854 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=12, min_samples_split=7;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=12, min_samples_split=8;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=12, min_samples_split=8;, score=0.871 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=12, min_samples_split=8;, score=0.858 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=12, min_samples_split=8;, score=0.854 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=12, min_samples_split=8;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=12, min_samples_split=9;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=12, min_samples_split=9;, score=0.871 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=12, min_samples_split=9;, score=0.858 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=12, min_samples_split=9;, score=0.854 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=12, min_samples_split=9;, score=0.849 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=13, min_samples_split=2;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=13, min_samples_split=2;, score=0.871 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=13, min_samples_split=2;, score=0.858 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=13, min_samples_split=2;, score=0.839 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=13, min_samples_split=2;, score=0.859 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=13, min_samples_split=3;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=13, min_samples_split=3;, score=0.871 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=13, min_samples_split=3;, score=0.858 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=13, min_samples_split=3;, score=0.839 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=13, min_samples_split=3;, score=0.859 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=13, min_samples_split=4;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=13, min_samples_split=4;, score=0.871 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=13, min_samples_split=4;, score=0.858 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=13, min_samples_split=4;, score=0.839 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=13, min_samples_split=4;, score=0.859 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=13, min_samples_split=5;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=13, min_samples_split=5;, score=0.871 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=13, min_samples_split=5;, score=0.858 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=13, min_samples_split=5;, score=0.839 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=13, min_samples_split=5;, score=0.859 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=13, min_samples_split=6;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=13, min_samples_split=6;, score=0.871 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=13, min_samples_split=6;, score=0.858 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=13, min_samples_split=6;, score=0.839 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=13, min_samples_split=6;, score=0.859 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=13, min_samples_split=7;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=13, min_samples_split=7;, score=0.871 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=13, min_samples_split=7;, score=0.858 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=13, min_samples_split=7;, score=0.839 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=13, min_samples_split=7;, score=0.859 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=13, min_samples_split=8;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=13, min_samples_split=8;, score=0.871 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=13, min_samples_split=8;, score=0.858 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=13, min_samples_split=8;, score=0.839 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=13, min_samples_split=8;, score=0.859 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=13, min_samples_split=9;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=13, min_samples_split=9;, score=0.871 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=13, min_samples_split=9;, score=0.858 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=13, min_samples_split=9;, score=0.839 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=13, min_samples_split=9;, score=0.859 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=14, min_samples_split=2;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=14, min_samples_split=2;, score=0.871 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=14, min_samples_split=2;, score=0.858 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=14, min_samples_split=2;, score=0.836 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=14, min_samples_split=2;, score=0.859 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=14, min_samples_split=3;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=14, min_samples_split=3;, score=0.871 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=14, min_samples_split=3;, score=0.858 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=14, min_samples_split=3;, score=0.836 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=14, min_samples_split=3;, score=0.859 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=14, min_samples_split=4;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=14, min_samples_split=4;, score=0.871 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=14, min_samples_split=4;, score=0.858 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=14, min_samples_split=4;, score=0.836 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=14, min_samples_split=4;, score=0.859 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=14, min_samples_split=5;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=14, min_samples_split=5;, score=0.871 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=14, min_samples_split=5;, score=0.858 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=14, min_samples_split=5;, score=0.836 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=14, min_samples_split=5;, score=0.859 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=14, min_samples_split=6;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=14, min_samples_split=6;, score=0.871 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=14, min_samples_split=6;, score=0.858 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=14, min_samples_split=6;, score=0.836 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=14, min_samples_split=6;, score=0.859 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=14, min_samples_split=7;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=14, min_samples_split=7;, score=0.871 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=14, min_samples_split=7;, score=0.858 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=14, min_samples_split=7;, score=0.836 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=14, min_samples_split=7;, score=0.859 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=14, min_samples_split=8;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=14, min_samples_split=8;, score=0.871 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=14, min_samples_split=8;, score=0.858 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=14, min_samples_split=8;, score=0.836 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=14, min_samples_split=8;, score=0.859 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=14, min_samples_split=9;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=14, min_samples_split=9;, score=0.871 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=14, min_samples_split=9;, score=0.858 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=14, min_samples_split=9;, score=0.836 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=14, min_samples_split=9;, score=0.859 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=15, min_samples_split=2;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=15, min_samples_split=2;, score=0.873 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=15, min_samples_split=2;, score=0.858 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=15, min_samples_split=2;, score=0.836 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=15, min_samples_split=2;, score=0.856 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=15, min_samples_split=3;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=15, min_samples_split=3;, score=0.873 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=15, min_samples_split=3;, score=0.858 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=15, min_samples_split=3;, score=0.836 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=15, min_samples_split=3;, score=0.856 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=15, min_samples_split=4;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=15, min_samples_split=4;, score=0.873 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=15, min_samples_split=4;, score=0.858 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=15, min_samples_split=4;, score=0.836 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=15, min_samples_split=4;, score=0.856 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=15, min_samples_split=5;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=15, min_samples_split=5;, score=0.873 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=15, min_samples_split=5;, score=0.858 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=15, min_samples_split=5;, score=0.836 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=15, min_samples_split=5;, score=0.856 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=15, min_samples_split=6;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=15, min_samples_split=6;, score=0.873 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=15, min_samples_split=6;, score=0.858 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=15, min_samples_split=6;, score=0.836 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=15, min_samples_split=6;, score=0.856 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=15, min_samples_split=7;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=15, min_samples_split=7;, score=0.873 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=15, min_samples_split=7;, score=0.858 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=15, min_samples_split=7;, score=0.836 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=15, min_samples_split=7;, score=0.856 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=15, min_samples_split=8;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=15, min_samples_split=8;, score=0.873 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=15, min_samples_split=8;, score=0.858 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=15, min_samples_split=8;, score=0.836 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=15, min_samples_split=8;, score=0.856 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=15, min_samples_split=9;, score=0.836 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=15, min_samples_split=9;, score=0.873 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=15, min_samples_split=9;, score=0.858 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=15, min_samples_split=9;, score=0.836 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=15, min_samples_split=9;, score=0.856 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=16, min_samples_split=2;, score=0.838 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=16, min_samples_split=2;, score=0.873 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=16, min_samples_split=2;, score=0.858 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=16, min_samples_split=2;, score=0.836 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=16, min_samples_split=2;, score=0.856 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=16, min_samples_split=3;, score=0.838 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=16, min_samples_split=3;, score=0.873 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=16, min_samples_split=3;, score=0.858 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=16, min_samples_split=3;, score=0.836 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=16, min_samples_split=3;, score=0.856 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=16, min_samples_split=4;, score=0.838 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=16, min_samples_split=4;, score=0.873 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=16, min_samples_split=4;, score=0.858 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=16, min_samples_split=4;, score=0.836 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=16, min_samples_split=4;, score=0.856 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=16, min_samples_split=5;, score=0.838 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=16, min_samples_split=5;, score=0.873 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=16, min_samples_split=5;, score=0.858 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=16, min_samples_split=5;, score=0.836 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=16, min_samples_split=5;, score=0.856 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=16, min_samples_split=6;, score=0.838 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=16, min_samples_split=6;, score=0.873 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=16, min_samples_split=6;, score=0.858 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=16, min_samples_split=6;, score=0.836 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=16, min_samples_split=6;, score=0.856 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=16, min_samples_split=7;, score=0.838 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=16, min_samples_split=7;, score=0.873 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=16, min_samples_split=7;, score=0.858 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=16, min_samples_split=7;, score=0.836 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=16, min_samples_split=7;, score=0.856 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=16, min_samples_split=8;, score=0.838 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=16, min_samples_split=8;, score=0.873 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=16, min_samples_split=8;, score=0.858 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=16, min_samples_split=8;, score=0.836 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=16, min_samples_split=8;, score=0.856 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=16, min_samples_split=9;, score=0.838 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=16, min_samples_split=9;, score=0.873 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=16, min_samples_split=9;, score=0.858 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=16, min_samples_split=9;, score=0.836 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=16, min_samples_split=9;, score=0.856 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=17, min_samples_split=2;, score=0.849 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=17, min_samples_split=2;, score=0.873 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=17, min_samples_split=2;, score=0.856 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=17, min_samples_split=2;, score=0.836 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=17, min_samples_split=2;, score=0.854 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=17, min_samples_split=3;, score=0.849 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=17, min_samples_split=3;, score=0.873 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=17, min_samples_split=3;, score=0.856 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=17, min_samples_split=3;, score=0.836 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=17, min_samples_split=3;, score=0.854 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=17, min_samples_split=4;, score=0.849 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=17, min_samples_split=4;, score=0.873 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=17, min_samples_split=4;, score=0.856 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=17, min_samples_split=4;, score=0.836 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=17, min_samples_split=4;, score=0.854 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=17, min_samples_split=5;, score=0.849 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=17, min_samples_split=5;, score=0.873 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=17, min_samples_split=5;, score=0.856 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=17, min_samples_split=5;, score=0.836 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=17, min_samples_split=5;, score=0.854 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=17, min_samples_split=6;, score=0.849 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=17, min_samples_split=6;, score=0.873 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=17, min_samples_split=6;, score=0.856 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=17, min_samples_split=6;, score=0.836 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=17, min_samples_split=6;, score=0.854 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=17, min_samples_split=7;, score=0.849 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=17, min_samples_split=7;, score=0.873 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=17, min_samples_split=7;, score=0.856 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=17, min_samples_split=7;, score=0.836 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=17, min_samples_split=7;, score=0.854 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=17, min_samples_split=8;, score=0.849 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=17, min_samples_split=8;, score=0.873 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=17, min_samples_split=8;, score=0.856 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=17, min_samples_split=8;, score=0.836 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=17, min_samples_split=8;, score=0.854 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=17, min_samples_split=9;, score=0.849 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=17, min_samples_split=9;, score=0.873 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=17, min_samples_split=9;, score=0.856 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=17, min_samples_split=9;, score=0.836 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=17, min_samples_split=9;, score=0.854 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=18, min_samples_split=2;, score=0.849 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=18, min_samples_split=2;, score=0.873 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=18, min_samples_split=2;, score=0.857 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=18, min_samples_split=2;, score=0.845 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=18, min_samples_split=2;, score=0.854 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=18, min_samples_split=3;, score=0.849 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=18, min_samples_split=3;, score=0.873 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=18, min_samples_split=3;, score=0.857 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=18, min_samples_split=3;, score=0.845 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=18, min_samples_split=3;, score=0.854 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=18, min_samples_split=4;, score=0.849 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=18, min_samples_split=4;, score=0.873 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=18, min_samples_split=4;, score=0.857 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=18, min_samples_split=4;, score=0.845 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=18, min_samples_split=4;, score=0.854 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=18, min_samples_split=5;, score=0.849 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=18, min_samples_split=5;, score=0.873 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=18, min_samples_split=5;, score=0.857 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=18, min_samples_split=5;, score=0.845 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=18, min_samples_split=5;, score=0.854 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=18, min_samples_split=6;, score=0.849 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=18, min_samples_split=6;, score=0.873 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=18, min_samples_split=6;, score=0.857 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=18, min_samples_split=6;, score=0.845 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=18, min_samples_split=6;, score=0.854 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=18, min_samples_split=7;, score=0.849 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=18, min_samples_split=7;, score=0.873 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=18, min_samples_split=7;, score=0.857 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=18, min_samples_split=7;, score=0.845 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=18, min_samples_split=7;, score=0.854 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=18, min_samples_split=8;, score=0.849 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=18, min_samples_split=8;, score=0.873 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=18, min_samples_split=8;, score=0.857 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=18, min_samples_split=8;, score=0.845 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=18, min_samples_split=8;, score=0.854 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=18, min_samples_split=9;, score=0.849 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=18, min_samples_split=9;, score=0.873 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=18, min_samples_split=9;, score=0.857 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=18, min_samples_split=9;, score=0.845 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=18, min_samples_split=9;, score=0.854 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=19, min_samples_split=2;, score=0.850 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=19, min_samples_split=2;, score=0.872 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=19, min_samples_split=2;, score=0.861 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=19, min_samples_split=2;, score=0.845 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=19, min_samples_split=2;, score=0.854 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=19, min_samples_split=3;, score=0.850 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=19, min_samples_split=3;, score=0.872 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=19, min_samples_split=3;, score=0.861 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=19, min_samples_split=3;, score=0.845 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=19, min_samples_split=3;, score=0.854 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=19, min_samples_split=4;, score=0.850 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=19, min_samples_split=4;, score=0.872 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=19, min_samples_split=4;, score=0.861 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=19, min_samples_split=4;, score=0.845 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=19, min_samples_split=4;, score=0.854 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=19, min_samples_split=5;, score=0.850 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=19, min_samples_split=5;, score=0.872 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=19, min_samples_split=5;, score=0.861 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=19, min_samples_split=5;, score=0.845 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=19, min_samples_split=5;, score=0.854 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=19, min_samples_split=6;, score=0.850 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=19, min_samples_split=6;, score=0.872 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=19, min_samples_split=6;, score=0.861 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=19, min_samples_split=6;, score=0.845 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=19, min_samples_split=6;, score=0.854 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=19, min_samples_split=7;, score=0.850 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=19, min_samples_split=7;, score=0.872 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=19, min_samples_split=7;, score=0.861 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=19, min_samples_split=7;, score=0.845 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=19, min_samples_split=7;, score=0.854 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=19, min_samples_split=8;, score=0.850 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=19, min_samples_split=8;, score=0.872 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=19, min_samples_split=8;, score=0.861 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=19, min_samples_split=8;, score=0.845 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=19, min_samples_split=8;, score=0.854 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=19, min_samples_split=9;, score=0.850 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=19, min_samples_split=9;, score=0.872 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=19, min_samples_split=9;, score=0.861 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=19, min_samples_split=9;, score=0.845 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=19, min_samples_split=9;, score=0.854 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=20, min_samples_split=2;, score=0.852 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=20, min_samples_split=2;, score=0.872 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=20, min_samples_split=2;, score=0.861 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=20, min_samples_split=2;, score=0.854 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=20, min_samples_split=2;, score=0.854 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=20, min_samples_split=3;, score=0.852 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=20, min_samples_split=3;, score=0.872 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=20, min_samples_split=3;, score=0.861 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=20, min_samples_split=3;, score=0.854 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=20, min_samples_split=3;, score=0.854 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=20, min_samples_split=4;, score=0.852 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=20, min_samples_split=4;, score=0.872 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=20, min_samples_split=4;, score=0.861 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=20, min_samples_split=4;, score=0.854 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=20, min_samples_split=4;, score=0.854 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=20, min_samples_split=5;, score=0.852 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=20, min_samples_split=5;, score=0.872 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=20, min_samples_split=5;, score=0.861 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=20, min_samples_split=5;, score=0.854 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=20, min_samples_split=5;, score=0.854 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=20, min_samples_split=6;, score=0.852 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=20, min_samples_split=6;, score=0.872 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=20, min_samples_split=6;, score=0.861 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=20, min_samples_split=6;, score=0.854 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=20, min_samples_split=6;, score=0.854 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=20, min_samples_split=7;, score=0.852 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=20, min_samples_split=7;, score=0.872 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=20, min_samples_split=7;, score=0.861 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=20, min_samples_split=7;, score=0.854 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=20, min_samples_split=7;, score=0.854 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=20, min_samples_split=8;, score=0.852 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=20, min_samples_split=8;, score=0.872 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=20, min_samples_split=8;, score=0.861 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=20, min_samples_split=8;, score=0.854 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=20, min_samples_split=8;, score=0.854 total time=   0.0s
    [CV 1/5] END max_leaf_nodes=20, min_samples_split=9;, score=0.852 total time=   0.0s
    [CV 2/5] END max_leaf_nodes=20, min_samples_split=9;, score=0.872 total time=   0.0s
    [CV 3/5] END max_leaf_nodes=20, min_samples_split=9;, score=0.861 total time=   0.0s
    [CV 4/5] END max_leaf_nodes=20, min_samples_split=9;, score=0.854 total time=   0.0s
    [CV 5/5] END max_leaf_nodes=20, min_samples_split=9;, score=0.854 total time=   0.0s





    GridSearchCV(cv=5, estimator=DecisionTreeClassifier(random_state=42),
                 param_grid={'max_leaf_nodes': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                                13, 14, 15, 16, 17, 18, 19, 20],
                             'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9]},
                 scoring='accuracy', verbose=3)



### Exercise 7d

Train it on the full training set using these hyperparameters, and measure your model's performance on the test set. You should get roughly 85% to 87% accuracy.


```python
from sklearn.metrics import accuracy_score

preds = gs.best_estimator_.predict(X_test)
print(accuracy_score(y_test, preds))
# Nice :)
```

    0.87


## Exercise 8
Grow a forest.

### Exercise 8a

Continuing the previous exercise, generate 1,000 subsets of the training set, each containing 100 instances selected randomly. Hint: you can use Scikit-learn's ShuffleSplit class for this.


```python
from sklearn.model_selection import ShuffleSplit

rs = ShuffleSplit(n_splits=1000, train_size=100, random_state=42)
forests = []
forests_y = []
for train_index, _ in rs.split(X_train):
    forests.append(X_train[train_index])
    forests_y.append(y_train[train_index])
print(len(forests))
print(len(forests[0]))
```

    1000
    100


### Exercise 8b

Train one Decision Tree on each subset, using the best hyperparameter values found above. Evaluate these 1,000 Decision Trees on the test set. Since they were trained on smaller sets these Decision Trees will likely perform worse than the first Decision Tree, achieving only about 80% accuracy.


```python
import numpy as np

best_params = gs.best_params_
scores = []
trees = []

for forest, forest_y in zip(forests, forests_y):
    forest_clf = DecisionTreeClassifier(**best_params, random_state=42)
    forest_clf.fit(forest, forest_y)
    trees.append(forest_clf)
    preds = forest_clf.predict(X_test)
    s = accuracy_score(y_test, preds)
    scores.append(s)
print(scores[:100])
print(np.mean(scores))
```

    [0.798, 0.836, 0.799, 0.8275, 0.7935, 0.8315, 0.7825, 0.7935, 0.784, 0.829, 0.794, 0.78, 0.781, 0.8135, 0.804, 0.846, 0.8255, 0.811, 0.825, 0.833, 0.776, 0.82, 0.792, 0.8165, 0.8155, 0.778, 0.833, 0.809, 0.8085, 0.809, 0.777, 0.831, 0.815, 0.77, 0.7845, 0.7925, 0.7905, 0.7665, 0.816, 0.8375, 0.7845, 0.728, 0.797, 0.792, 0.7895, 0.791, 0.813, 0.7835, 0.75, 0.7985, 0.8185, 0.817, 0.7805, 0.793, 0.781, 0.8045, 0.8215, 0.818, 0.765, 0.7565, 0.771, 0.8035, 0.7665, 0.8085, 0.8205, 0.7775, 0.822, 0.781, 0.838, 0.759, 0.8215, 0.803, 0.806, 0.8205, 0.8165, 0.769, 0.831, 0.801, 0.831, 0.7915, 0.797, 0.7875, 0.848, 0.8095, 0.735, 0.7425, 0.808, 0.8035, 0.819, 0.7255, 0.7945, 0.779, 0.7945, 0.7865, 0.7915, 0.792, 0.816, 0.801, 0.818, 0.79]
    0.8012485


### Exercise 8c

Now comes the magic(!). For each test set instance, generate the predictions of the 1,000 Decision Trees, and keep only the most frequent prediction (you can use SciPy's `mode()` function for this). This gives you the _majority-vote predictions_ over the test set.


```python
from scipy.stats import mode

c = {k: [] for k in range(len(X_test))}

for tree in trees:
    for ind, pred in zip(range(len(X_test)), tree.predict(X_test)):
        c[ind].append(pred)

for k, v in c.items():
    m = mode(v)[0][0]
    c[k] = m
```

### Exercise 8d

Evaluate these predictions on the test set: you should obtain a slightly higher accuracy than your first model (about 0.5 to 1.5% higher). Congratulations, you have trained a Random Forest classifier!


```python
print(accuracy_score(y_test, np.array(list(c.values()))))
```

    0.872


Note quite the increase we were expecting, but that's what Geron got... my first model was higher at 87% already.
