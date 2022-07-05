---
layout: exercises
chapter: 7
chapter-title: Ensemble Learning and Random Forests
permalink: /ml-book/chapter7/exercises.html
---

## Exercise 1
If I've trained five different models on the exact same training data and they all achieve 95% precision there is a chance that I could combine these models to get better results. You would combine the 5 predictions and predict the majority vote and that could achieve a precision higher than 95%.

Accuracy: (TP + TN) / (FP + FN + TP + TN)
Precision: TP / (TP + FP)

## Exercise 2
A hard vote counts the weight of each vote equally when determining the majority vote, whereas a soft voting classifier will weight each vote according to it's `predict_proba` value.

## Exercise 3
Bagging is when you sample with replacement, so it would definitely be possible to distribute across multiple servers, you just need to copy all the data to each server. I think it would also be possible to distribute pasting ensembles across several servers, again just put a copy of the data on each server and sample without replacement. Boosting sounds difficult to spread across servers because each model needs to correct it's predecessor. Yes random forests are possible to distribute. Stacking would be partly possible since the parts before the blender could be distributed.

## Exercise 4
Out-of-bag evaluation allows you to gain an idea of how a bagging algorithm will perform on new data, because roughly 37% of data will not have been used in training the model. Therefore, this portion of the training sample can be used for evaluation much like a validation set.

## Exercise 5
Extra-Trees are more random than Random Forests because instead of choosing the optimal feature/threshold at each node (i.e. the feature (amongst a random subset) and threshold which best splits the data at the node), it randomly selects a threshold for each feature. Therefore, the resultant tree is less predictable since each split was made randomly based on these thresholds. They're faster than Random Forests since they don't require time to compute the optimal feature/threshold for each node.

## Exercise 6
If your AdaBoost ensemble underfits the training data you should increase the learning rate (defaults to 1). You can also try increasing the number of estimators or decreasing the default regularization parameters.

## Exercise 7
If your Gradient Boosting ensemble overfits then you should decrease the learning rate so that it will generalize better.

## Exercise 8
Load the MNIST data and split it into a training set, a validation set, and a test set (50k (typo in book), 10k, 10k). Then train various classifiers, such as a Random Forest classifier, an Extra-Trees classifier, and an SVM. Next, try to combine them into an ensemble that outperforms them all on the validation set, using a soft or hard voting classifier. Once you have found one try it on the test set. How much better does it perform compared to the individual classifiers?


```python
# From chapter 3:
from sklearn.datasets import fetch_openml
import numpy as np

# MNIST changed to https://www.openml.org/d/554
mnist = fetch_openml("mnist_784", version=1, as_frame=False)

```


```python
from sklearn.model_selection import train_test_split

X, y = mnist["data"], mnist["target"]

X_train, X_inter, y_train, y_inter = train_test_split(X, y, test_size=20_000)
X_val, X_test, y_val, y_test = train_test_split(X_inter, y_inter, test_size=10_000)

print(X_train.shape, X_val.shape, X_test.shape)
```

    (50000, 784) (10000, 784) (10000, 784)



```python
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC

rfc, etc, lsvc = (
    RandomForestClassifier(random_state=42),
    ExtraTreesClassifier(random_state=42),
    LinearSVC(random_state=42),
)
```


```python
rfc.fit(X_train, y_train)
print("rfc done")
etc.fit(X_train, y_train)
print("etc done")
lsvc.fit(X_train, y_train)
print("lsvc done")

```

    rfc done
    etc done
    lsvc done


    /Users/riley/PycharmProjects/ML/venv/lib/python3.8/site-packages/sklearn/svm/_base.py:1206: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      warnings.warn(



```python
from sklearn.svm import SVC

svc = SVC(tol=1, random_state=42)
svc.fit(X_train, y_train)
```




    SVC(random_state=42, tol=1)




```python
from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier(n_estimators=100, random_state=42)
ada.fit(X_train, y_train)
```




    AdaBoostClassifier(n_estimators=100, random_state=42)




```python
models = [
    ("rfc", rfc),
    ("etf", etc),
    ("svc", svc),
    ("ada", ada),
]
```


```python
for model in models:
    print(model[0], model[1].score(X_val, y_val))
```

    rfc 0.9667
    etf 0.9696
    svc 0.9784
    ada 0.7274



```python
from sklearn.ensemble import VotingClassifier

vc = VotingClassifier(models, voting="hard", n_jobs=6)
vc.fit(X_train, y_train)
vc.score(X_val, y_val)
```




    0.9697




```python
# Let's remove ada
models = [
    ("rfc", rfc),
    ("etf", etc),
    ("svc", svc),
]
vc = VotingClassifier(models, voting="hard", n_jobs=6)
vc.fit(X_train, y_train)
vc.score(X_val, y_val)
```




    0.9737




```python
# Still not better than the SVM let's try the soft
svc.probability = True
models[2][1].probability = True
vc_soft = VotingClassifier(models, voting="soft", n_jobs=6)
vc_soft.fit(X_train, y_train)
vc_soft.score(X_val, y_val)
```




    0.9792




```python
# So we did better, let's try on test set
print("VC Soft Score:", vc_soft.score(X_test, y_test))
for model in vc_soft.estimators_:
    print(model.__class__.__name__ + ":", model.score(X_test, y_test))
```

    VC Soft Score: 0.9767
    RandomForestClassifier: 0.9647
    ExtraTreesClassifier: 0.9676
    SVC: 0.976


So we did a very little bit better. Our soft voting classifier had 97.67% accuracy and our SVC alone had 97.60% accuracy.

## Exercise 9
Run the individual classifiers from the previous exercise to make predictions on the validation set, and create a new training set with the resulting predictions: each training instance is a vector containing the set of predictions from all your classifiers for an image, and the target is the image's class. Congratulations, you have just trained a blender, and together with the classifiers they form a stacking ensemble! Now let's evaluate the ensemble on the test set. For each image in the test set, make predictions with all your classifiers, then feed the predictions to the blender to get the ensemble's predictions. How does it compare to the voting classifier you trained earlier?


```python
X_new = -np.ones((X_val.shape[0], 10*len(vc_soft.estimators_)))
X_new[:, :10] = vc_soft.estimators_[0].predict_proba(X_val)
X_new[:, 10:20] = vc_soft.estimators_[1].predict_proba(X_val)
X_new[:, 20:] = vc_soft.estimators_[2].predict_proba(X_val)
y_new = y_val
print(X_new)
print(X_new.shape, y_new.shape)
```

    [[9.10000000e-01 0.00000000e+00 0.00000000e+00 ... 7.79371387e-07
      1.09459623e-06 3.72442020e-06]
     [9.50000000e-01 0.00000000e+00 1.00000000e-02 ... 4.23279470e-05
      5.95567083e-05 1.67694188e-03]
     [8.70000000e-01 0.00000000e+00 0.00000000e+00 ... 1.10429014e-03
      4.04542473e-05 2.90887552e-04]
     ...
     [0.00000000e+00 1.00000000e-02 3.00000000e-02 ... 4.31194473e-05
      1.44719826e-03 1.08973280e-04]
     [2.00000000e-02 0.00000000e+00 6.00000000e-02 ... 2.61564119e-04
      1.06890698e-03 2.14030781e-04]
     [3.00000000e-02 1.00000000e-02 2.00000000e-02 ... 1.05278615e-02
      6.03766703e-03 5.01682195e-01]]
    (10000, 30) (10000,)



```python
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(max_iter=1000)
mlp.fit(X_new, y_new)
mlp.score(X_new, y_new)
```




    0.994




```python
X_new_test = -np.ones((X_test.shape[0], 10*len(vc_soft.estimators_)))
X_new_test[:, :10] = vc_soft.estimators_[0].predict_proba(X_test)
X_new_test[:, 10:20] = vc_soft.estimators_[1].predict_proba(X_test)
X_new_test[:, 20:] = vc_soft.estimators_[2].predict_proba(X_test)
y_new_test = y_test
print(X_new_test)
print(X_new_test.shape, y_new_test.shape)
```

    [[0.00000000e+00 0.00000000e+00 0.00000000e+00 ... 1.17529491e-09
      9.99998859e-01 1.11848133e-06]
     [4.00000000e-02 1.70000000e-01 4.00000000e-02 ... 9.16153445e-03
      5.74427818e-01 9.49229029e-03]
     [0.00000000e+00 0.00000000e+00 3.00000000e-02 ... 9.79965813e-01
      4.73102141e-04 6.88522020e-03]
     ...
     [1.00000000e-02 4.00000000e-02 1.10000000e-01 ... 1.12631510e-05
      9.98781514e-01 4.08627716e-04]
     [0.00000000e+00 3.40000000e-01 0.00000000e+00 ... 4.80205378e-01
      8.56538302e-03 7.77163413e-02]
     [1.80000000e-01 0.00000000e+00 3.00000000e-02 ... 6.09119548e-03
      6.08639737e-02 8.56811451e-03]]
    (10000, 30) (10000,)



```python
mlp.score(X_new_test, y_new_test)
```




    0.9746



So we did a very little bit worse with the stacking ensemble. Our soft voting classifier had 97.67% accuracy and our stacking ensemble had 97.46% accuracy on the test set.
