---
layout: exercises
chapter: 4
chapter-title: Training Models
permalink: /ml-book/chapter4/exercises.html
---

## Exercise 1

You can use any flavor of Gradient Descent (Batch (provided there are few enough samples to fit in memory), Stochastic, or Mini-batch) if you have a training set with millions of features.

## Exercise 2

All forms of Gradient Descent might suffer from features with very different scales. This is because they will converge slower since the gradient descent will take a circuitous route to the minimum as 4-7 illustrates. I like to think about this as an ellipse and the gradient descent goes towards the semi major axis first, once it reaches there it heads along it towards the minimum. To fix this you can standard scale or min max scale the features.

## Exercise 3

No, the log loss function for Logistic Regression is convex so there's no need to worry about it getting stuck in a local minimum.

## Exercise 4

Not all Gradient Descent algorithms lead to the exact same model provided you let them run long enough. Most of the time it's close enough, but Batch Gradient descent will converge smoothly to a specific minimum. However, Stochastic Gradient Descent and Mini-Batch Gradient Descent may "bounce" around the global minimum. However, if you gradually lower the learning rate they will become closer and closer to BGD.

## Exercise 5

If the validation error goes up at every epoch you're likely overfitting. There are various methods to prevent overfitting including: Use a less complicated model, use Ridge, Lasso, or Elastic Net regularization, increase the size of your (training) dataset. Geron notes that it could also be that the learning rate is too high this would definitely be the case if the training error is going up as well.

## Exercise 6

Typically to implement early stopping you would have some patience factor. I don't think it would be a good idea to stop immediately because of this. For example, the validation error could be consistently going down, increase for one epoch, and then go down for the next 10 epochs. The patience factor says something like "if _ epochs go by without a decrease in the validation error, then stop (and revert to minimum validation error model)".

## Exercise 7

The normal equations will be fast when the number of features is low because it is linear in the number of samples, it will converge exactly. Batch Gradient Descent will be slow for a large number of samples, but largely unaffected by the number of features, it will converge exactly. Stochastic Gradient Descent and Mini-batch Gradient Descent will both be fast, but require that the learning rate be decreases so that they actually converge.

## Exercise 8

Three ways to solve a gap between the training error and validation error in Polynomial Regression are: Increase size of the dataset, apply regularization, use a less complicated model. Overfitting is what is happening. Nailed this answer : )

## Exercise 9

If the training error and validation error are almost equal and fairly high then that indicates high bias. You should reduce the regularization parameter $\alpha$.

## Exercise 10

* Ridge Regression instead of Linear Regression?
  * You want to prevent overfitting (high variance) in your model.
* Lasso instead of Ridge Regression?
  * You want to completely eliminate the impact of the least important features instead of just penalizing them.
* Elastic Net instead of Lasso?
  * You want to reduce the complexity of your model and only use the most important features, but want to avoid the erratic behavior of Lasso (when # features > # samples or several features are strongly correlated).
  * Good tip from Geron: If you want to use Lasso without the negative effects just use Elastic Net with an l1 ratio close to 1.

## Exercise 11

If you want to classify pictures as outdoor/indoor and daytime/nighttime, then you should implement two Logistic Regression classifiers instead of one Softmax Regression classifier. Softmax Regression is multi-class, not multi-output.

## Exercise 12

Implement Batch Gradient Descent with early stopping for Softmax Regression


```python
# This sounds difficult, but let's give it a shot
from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()
```


```python
X, y = iris["data"], iris["target"]
arr = np.hstack((X, y[np.newaxis, :].T))
np.random.seed(5)
p = np.random.permutation(len(arr))
arr = arr[p]
s = (np.array([0.7, 0.2, 0.1]) * len(arr)).astype(int)
s = np.cumsum(s)
print(s)
X_train, y_train = arr[0:s[0], :-1], arr[0:s[0], -1][:, np.newaxis]
print(X_train.shape, y_train.shape)
X_val, y_val = arr[s[0]:s[1], :-1], arr[s[0]:s[1], -1][:, np.newaxis]
print(X_val.shape, y_val.shape)
X_test, y_test = arr[s[1]:s[2], :-1], arr[s[1]:s[2], -1][:, np.newaxis]
print(X_test.shape, y_test.shape)
```

    [105 135 150]
    (105, 4) (105, 1)
    (30, 4) (30, 1)
    (15, 4) (15, 1)



```python
def softmax_eval(X, y, theta):
    y = y[:, 0]
    m = X.shape[0]
    K = len(set(y))
    l = 0
    for k in range(K):
        y_tmp = (y == k).astype(int)
        for i in range(m):
            s_k = theta[:, k].T @ X[i]
            p_k = np.exp(s_k) / sum(np.exp(theta[:, k].T @ X[i]) for k in range(K))
            l += y_tmp[i] * np.log(p_k)
    l = (-1 / m) * l
    return l
        

def train(X, y, iters=1000, lr=0.01):
    y = y[:, 0]
    m = X.shape[0]
    K = len(set(y))
    theta = np.ones((X.shape[1], K))
    eta = lr
    for iteration in range(iters):
        for k in range(K):
            y_tmp = (y == k).astype(int)
            grad_k = 0
            for i in range(m):
                s_k = theta[:, k].T.dot(X[i])
                p_k = np.exp(s_k) / sum(np.exp(theta[:, k].T @ X[i]) for k in range(K))
                grad_k += (p_k - y_tmp[i]) * X[i]
            grad_k = (1 / m)*grad_k
            theta[:, k] = theta[:, k] - eta * grad_k
        if (iteration + 1) % 100 == 0:
            l = softmax_eval(X, y.reshape((-1, 1)), theta)
            print("Loss: ", round(l, 2))
    return theta

train(X_train, y_train)
```

    Loss:  0.73
    Loss:  0.6
    Loss:  0.54
    Loss:  0.5
    Loss:  0.47
    Loss:  0.44
    Loss:  0.42
    Loss:  0.41
    Loss:  0.39
    Loss:  0.38





    array([[ 1.3326466 ,  1.17699356,  0.49879875],
           [ 1.84638559,  0.75029511,  0.40886567],
           [-0.19686822,  1.2119394 ,  1.98789651],
           [ 0.44451838,  0.84531752,  1.71079481]])



This was my first attempt... looking at Geron's work there's still a little to do here.
The main difference is that he vectorized everything. He also added the bias term which I forgot to do.


```python
def one_hot(y):
    K = len(set(y[:, 0]))
    new_y = np.zeros((len(y), K))
    i, j = np.indices(new_y.shape)
    return (j == y).astype(int)
```


```python
def cross_entropy_loss(X, y, theta):
    m = X.shape[0]
    # Assume X has bias and y is onehot at this point
    s = X.dot(theta)
    p = np.exp(s) / np.sum(np.exp(s), axis=1)[:, np.newaxis]
    return (-1 / m) * np.sum(y * np.log(p))

def accuracy(X, y, theta):
    s = X.dot(theta)
    p = np.exp(s) / np.sum(np.exp(s), axis=1)[:, np.newaxis]
    pmax = np.argmax(p, axis=1)
    preds = y[range(y.shape[0]), pmax]
    return sum(preds) / sum(np.ones(len(preds)))
    


def train_vectorized(X, y, X_val, y_val, iters=1000001, lr=0.01, patience=5):
    # Add bias to X
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    X_val = np.hstack((np.ones((X_val.shape[0], 1)), X_val))
    
    # One hot y
    y = one_hot(y)
    y_val = one_hot(y_val)
    
    
    m = X.shape[0]
    K = y.shape[1]

    # Initial guess for theta
    theta = np.ones((X.shape[1], K))
    eta = lr
    best_val_loss = np.inf
    
    for iteration in range(iters):
        # s_k = theta_k^T \cdot x
        s = X.dot(theta)
        p = np.exp(s) / np.sum(np.exp(s), axis=1)[:, np.newaxis]
        grad = (1 / m) * X.T.dot(p - y)
        theta -= eta * grad

        val_loss = cross_entropy_loss(X_val, y_val, theta)
        if ((iteration + 1) % 1000) == 0:
            print(iteration + 1, val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_theta = theta
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print("Early Stopping")
                return best_theta, best_val_loss
    print("Warning no Early Stopping")
    return best_theta, best_val_loss
            
        
theta, _ = train_vectorized(X_train, y_train, X_val, y_val)

```

    1000 0.3342596544930866
    2000 0.25429893944101895
    3000 0.21449143442540694
    4000 0.19125970341767579
    5000 0.17637229776556052
    6000 0.16619324074143058
    7000 0.15889600701967801
    8000 0.1534742080223439
    9000 0.1493327048861036
    10000 0.14609930897339848
    11000 0.1435304671922347
    12000 0.14146072082146635
    13000 0.13977407415470203
    14000 0.1383870025727842
    15000 0.13723796172201708
    16000 0.136280686463588
    17000 0.13547978213770803
    18000 0.1348077477661006
    19000 0.1342429194633856
    20000 0.13376802026256793
    21000 0.1333691186414038
    22000 0.13303486811330995
    23000 0.13275594366356386
    24000 0.1325246183485923
    25000 0.13233444121623
    26000 0.13217998948842136
    27000 0.1320566758700738
    28000 0.13196059726122655
    29000 0.13188841490471628
    30000 0.1318372586424963
    31000 0.13180464983522938
    32000 0.1317884388562891
    Early Stopping



```python
# Add bias to X
X_test = np.hstack((np.ones((X_test.shape[0], 1)),X_test))

# One hot y
y_test = one_hot(y_test)
accuracy(X_test, y_test, theta)
```




    1.0




```python
s = X_test.dot(theta)
p = np.exp(s) / np.sum(np.exp(s), axis=1)[:, np.newaxis]
for i in range(len(p)):
    print(np.round(p[i], 3), y_test[i])
```

    [0.    0.124 0.876] [0 0 1]
    [0.001 0.983 0.015] [0 1 0]
    [0.98 0.02 0.  ] [1 0 0]
    [0.004 0.994 0.002] [0 1 0]
    [0.992 0.008 0.   ] [1 0 0]
    [0.996 0.004 0.   ] [1 0 0]
    [0.003 0.986 0.011] [0 1 0]
    [0.    0.004 0.996] [0 0 1]
    [0.994 0.006 0.   ] [1 0 0]
    [0.    0.038 0.962] [0 0 1]
    [0.98 0.02 0.  ] [1 0 0]
    [0.    0.948 0.051] [0 1 0]
    [0.    0.003 0.997] [0 0 1]
    [0. 0. 1.] [0 0 1]
    [0.003 0.979 0.018] [0 1 0]


I had an error in my code where the variable m in the cross_entropy_loss was being taken from the global context (since it wasn't defined in the scope of the function). As a result, the validation loss was constantly decreasing. That was pretty annoying because I was like how am I supposed to implement early stopping if it just keeps decreasing. Anyways, I'm glad I got this working eventually :). One thing I learned is how to convert from a sum to a vectorized form in code. Here's a quick demo of that.


```python
X_train_bias = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
print(
    "X shape:",
    X_train_bias.shape,
    "\ny shape:",
    y_train.shape,
    "\n" "theta shape:",
    theta.shape,
)

# Number of features in the training set should be equal to the number of fitted parameters in theta
assert X_train_bias.shape[1] == theta.shape[0]

# Number of classes should be equal to the number of columns in theta
assert len(set(y_train[:, 0])) == theta.shape[1]


def longway(X, y, theta):
    dtheta = np.zeros(theta.shape)
    m = len(X)
    J = theta.shape[0]
    K = len(set(y_train[:, 0]))
    for k in range(K):
        for j in range(J):
            s = 0
            for i in range(m):
                s += (theta[:, k].dot(X[i]) - y[i, 0]) * X[i, j]
            dtheta[j, k] = (2 / m) * s
    return dtheta


def shortway(X, y, theta):
    m = len(X)
    J = theta.shape[0]
    K = len(set(y_train[:, 0]))
    X.dot(theta) - y
    return (2 / m) * X.T.dot(X.dot(theta) - y)
```

    X shape: (105, 5) 
    y shape: (105, 1) 
    theta shape: (5, 3)



```python
# These are equivalent methods
# Obviously the vectorized one is easier to code and understand
# although it takes some thought to write it out

np.array_equal(shortway(X_train_bias, y_train, theta).round(4), 
               longway(X_train_bias, y_train, theta).round(4))
```




    True




```python
# And the shortway is much faster!!
%timeit longway(X_train_bias, y_train, theta)
%timeit shortway(X_train_bias, y_train, theta)
```

    2.69 ms ± 59.4 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    26.5 µs ± 165 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)



```python

```
