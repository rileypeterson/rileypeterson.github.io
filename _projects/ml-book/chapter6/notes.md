---
layout: notes
chapter: 6
chapter-title: Decision Trees
permalink: /ml-book/chapter6/notes.html
---

## Training and Visualizing a Decision
* They're a fundamental components of Random Forests


```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X = iris.data[:, 2:]
y = iris.target

tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y)
```




    DecisionTreeClassifier(max_depth=2)



![image](https://user-images.githubusercontent.com/29719483/173128779-52b93dbc-fa77-4765-9f19-c20c8b3afaa4.png)


## Making Predictions
* As the diagram shows first it splits the dataset based on whether or not `petal length <= 2.45`
  * If `True`, we reach the leaf node where the class is 100% setosa
  * Else we continue the the next questions which further separates the based on the petal width
* Decision Trees do not require feature scaling or centering at all
* Scikit-learn uses the CART algorithm which is a binary tree, but other algorithms (e.g. ID3) can produce trees with 3+ leaves

#### Gini Impurity
* Gini Impurity is a measure of how impure the node is
  * The impurity is zero when all training instances it applies to belong to the same class (e.g. Left most node above)

$$G_i = 1 - \sum_{k=1}^{n} p_{i,k}^2$$

* $p_{i,k}$ is the ratio of the class k instances among the training instance in the ith node.

Here is the decision boundary created by the tree:

![image](https://user-images.githubusercontent.com/29719483/173130302-13031246-b489-4d2d-a891-088bff582ae3.png)


* Decision Trees are considered a white box model (as opposed to black box) because the it's easy to retrace why decisions were made by the model, whereas in neural network that is not as feasible

## Estimating Class Probabilities
* It assigns probabilities simply based on the fraction of instances for each class that showed up in the associated node in the training set


```python
print(tree_clf.predict_proba([[5, 1.5]]))
print(tree_clf.predict([[5, 1.5]]))
```

    [[0.         0.90740741 0.09259259]]
    [1]


## The CART Training Algorithm
* The CART algorithm works as follows:
  * At the root node pick a feature and a threshold for that feature which splits the dataset in half
    * Pick the feature threshold combo which results in the purest subsets (weighted by their size)
  * Continue recursively down the tree until some user defined stop condition is met (i.e. max_depth) or it cannot find an additional split which will reduce the impurity.
* This is a Greedy algorithm in the sense that it starts with an optimal split, and continues to search for it at each level, but it doesn't consider the impurity of lower levels when making a decision for the current level
* In general the optimal tree is NP-Complete $O(exp(m))$ and intractable, in general the CART algorithm produces a reasonably good solution

## Computational Complexity
* It's essential a binary search tree which is $O(log(m))$ to traverse for predictions
* For training, since it compares all features on all samples at each node it is $O(n \times log(m))$

## Gini Impurity or Entropy?
* Entropy is another impurity measure
* A set's entropy is zero when it contains instances of only one class (which is also when gini is 0)

$$H_i = - \sum\limits_{\substack{k=1 \\ p_{i,k} \neq 0}}^n p_{i,k} log(p_{i,k})$$

* **Most of the time it doesn't matter which metric you use. Gini impurity is slightly faster to compute, so it's a good default. When they differ, Gini impurity tends to isolate the most frequent class in its own branch of the tree, while entropy tends to produce slightly more balanced trees.**

## Regularization Hyperparameters
* Decision trees left unconstrained are liable to overfit the training data
* They are known as nonparametric models because their parameters are not pre-defined such as in a linear model
* To prevent overfitting, we can use regularization to restrict the degrees of freedom of the decision tree
* There are several regularization parameters to pick from
* Increasing the `min` hyperparameters and reducing the `max` will regularize the model

* Other algorithms work by creating an unrestricted tree and then pruning unnecessary nodes
* Nodes are deemed unnecessary if the children of a node are both end leaves and the p-value dictates that the chances of that split are statiscally insignifcant

![image](https://user-images.githubusercontent.com/29719483/173135440-072bd762-ae06-40d7-9c9c-d56455ecf987.png)

## Regression
* Find an x value that splits the training targets in half and continue splitting until stopping condition, the target value is the mean of the target value of the samples in that node.
* I think this is kind of dumb... like when are results from this sort of regression going to be better... boundaries are only orthogonal

![image](https://user-images.githubusercontent.com/29719483/173137358-b3b5849c-2ea4-4f87-976e-2c6ca0af64dd.png)

## Instability
* Only orthogonal decision boundary which makes them susceptible to rotations in the training data
* Can use PCA to mitigate this issue
* Very sensitive to small variations in the training data
* Removing one sample from the Iris dataset results in a totally different tree, even without removing samples the resultant tree might be different just because of the stochastic nature of the Scikit-learn algorithm
* Random Forests can limit this instability by averaging predictions over many trees


```python

```
