---
layout: exercises
chapter: 2
chapter-title: End-to-End Machine Learning Project
permalink: /ml-book/chapter2/exercises.html
---

## Exercise 1

Try a Support Vector Machine with various hyperparameters. So that means grid search on the hyperparameters. We're going to start from scratch just to recall all the steps.


```python
import tarfile
import tempfile
import urllib.request
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

housing_url = (
    "https://raw.githubusercontent.com/ageron/"
    + "handson-ml/master/datasets/housing/housing.tgz"
)
FIGSIZE = (16, 12)


def read_tar(url):
    r = urllib.request.urlopen(url)
    with tempfile.TemporaryDirectory() as d:
        with tarfile.open(fileobj=r, mode="r:gz") as tf:
            tf.extractall(path=d)
            name = tf.getnames()[0]
        df = pd.read_csv(os.path.join(d, name))
    return df


df = read_tar(housing_url)

# Stratify by important features
strat_values = df["median_income"]
bins = 5
strat = np.ceil(strat_values / 1.5)
strat = strat.where(strat < 5, 5.0)
df["income_cat"] = strat
strat_train_set, strat_test_set = train_test_split(
    df, test_size=0.2, random_state=42, stratify=strat
)

# Remove strat feature
cols = [i for i in df.columns if i != "income_cat"]
df = df.loc[:, cols]
strat_train_set = strat_train_set.loc[:, cols]
strat_test_set = strat_test_set.loc[:, cols]

# Add combo features
df["population_per_household"] = df["population"] / df["households"]
df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
df["rooms_per_household"] = df["total_rooms"] / df["households"]

# Remove target from inputs
target = "median_house_value"
x = df[[col for col in df.columns if col != target]].copy()
y = df[[target]].copy()
```


```python
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Pipeline
cat_cols = ["ocean_proximity"]
num_cols = [col for col in x.columns if col not in cat_cols]

num_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("std_scaler", StandardScaler()),
    ]
)

pipeline = ColumnTransformer(
    [
        ("num", num_pipeline, num_cols),
        ("cat", OneHotEncoder(), cat_cols),
    ]
)
x_final = pipeline.fit_transform(x)
y_final = y.copy().values[:, 0]
print(x_final.shape)
print(y_final.shape)
```

    (20640, 16)
    (20640,)



```python
# Grid search
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR

param_grid = [
    {"kernel": ["linear"], "C": np.logspace(-2, 2, 5)},
    {"kernel": ["rbf"], "C": np.logspace(-2, 2, 5), "gamma": np.logspace(-3, 1, 5)},
]

svr_reg = SVR()

grid_search = GridSearchCV(
    svr_reg,
    param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    return_train_score=True,
    n_jobs=4,
    verbose=4,
)
grid_search.fit(x_final, y_final)
```

    Fitting 5 folds for each of 30 candidates, totalling 150 fits


    [Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=4)]: Done  17 tasks      | elapsed:  1.5min
    [Parallel(n_jobs=4)]: Done  90 tasks      | elapsed:  8.9min
    [Parallel(n_jobs=4)]: Done 150 out of 150 | elapsed: 15.1min finished





    GridSearchCV(cv=5, estimator=SVR(), n_jobs=4,
                 param_grid=[{'C': array([1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02]),
                              'kernel': ['linear']},
                             {'C': array([1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02]),
                              'gamma': array([1.e-03, 1.e-02, 1.e-01, 1.e+00, 1.e+01]),
                              'kernel': ['rbf']}],
                 return_train_score=True, scoring='neg_mean_squared_error',
                 verbose=4)




```python
print(grid_search.best_params_)
# Since these were the max we probably want to run it with higher values...
print(grid_search.best_estimator_)
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
```

    {'C': 100.0, 'kernel': 'linear'}
    SVR(C=100.0, kernel='linear')
    120046.83746066342 {'C': 0.01, 'kernel': 'linear'}
    119313.36974987865 {'C': 0.1, 'kernel': 'linear'}
    112692.04970886311 {'C': 1.0, 'kernel': 'linear'}
    84708.66839222891 {'C': 10.0, 'kernel': 'linear'}
    74449.9743277484 {'C': 100.0, 'kernel': 'linear'}
    120132.95001003973 {'C': 0.01, 'gamma': 0.001, 'kernel': 'rbf'}
    120131.66921718646 {'C': 0.01, 'gamma': 0.01, 'kernel': 'rbf'}
    120129.58678535324 {'C': 0.01, 'gamma': 0.1, 'kernel': 'rbf'}
    120132.74223046939 {'C': 0.01, 'gamma': 1.0, 'kernel': 'rbf'}
    120133.11221026107 {'C': 0.01, 'gamma': 10.0, 'kernel': 'rbf'}
    120131.32982041281 {'C': 0.1, 'gamma': 0.001, 'kernel': 'rbf'}
    120118.43736317645 {'C': 0.1, 'gamma': 0.01, 'kernel': 'rbf'}
    120097.61802146187 {'C': 0.1, 'gamma': 0.1, 'kernel': 'rbf'}
    120129.26313275466 {'C': 0.1, 'gamma': 1.0, 'kernel': 'rbf'}
    120133.07528895183 {'C': 0.1, 'gamma': 10.0, 'kernel': 'rbf'}
    120115.04323111309 {'C': 1.0, 'gamma': 0.001, 'kernel': 'rbf'}
    119995.12676569127 {'C': 1.0, 'gamma': 0.01, 'kernel': 'rbf'}
    119815.3873602283 {'C': 1.0, 'gamma': 0.1, 'kernel': 'rbf'}
    120095.48937602463 {'C': 1.0, 'gamma': 1.0, 'kernel': 'rbf'}
    120132.62190047713 {'C': 1.0, 'gamma': 10.0, 'kernel': 'rbf'}
    119965.05566501652 {'C': 10.0, 'gamma': 0.001, 'kernel': 'rbf'}
    118854.6745015883 {'C': 10.0, 'gamma': 0.01, 'kernel': 'rbf'}
    117089.72323801681 {'C': 10.0, 'gamma': 0.1, 'kernel': 'rbf'}
    119858.45250446712 {'C': 10.0, 'gamma': 1.0, 'kernel': 'rbf'}
    120129.00480115737 {'C': 10.0, 'gamma': 10.0, 'kernel': 'rbf'}
    118561.05702632615 {'C': 100.0, 'gamma': 0.001, 'kernel': 'rbf'}
    108868.53282659988 {'C': 100.0, 'gamma': 0.01, 'kernel': 'rbf'}
    99283.73122290114 {'C': 100.0, 'gamma': 0.1, 'kernel': 'rbf'}
    117585.44385508774 {'C': 100.0, 'gamma': 1.0, 'kernel': 'rbf'}
    120101.26502369305 {'C': 100.0, 'gamma': 10.0, 'kernel': 'rbf'}



```python
# Best one is: 74449.9743277484 {'C': 100.0, 'kernel': 'linear'}
```

## Exercise 2
Try `RandomizedSearchCV` instead of the grid search.


```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import recipinvgauss, reciprocal

# Since C is centered at

param_grid = {
    "kernel": ["linear", "rbf"],
    "C": reciprocal(1, 1000000),
    "gamma": recipinvgauss(mu=1, scale=0.5),
}


svr_reg = SVR()

grid_search = RandomizedSearchCV(
    svr_reg,
    param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    return_train_score=True,
    n_jobs=4,
    n_iter=30,
    verbose=4,
)
grid_search.fit(x_final, y_final)
```

    Fitting 5 folds for each of 30 candidates, totalling 150 fits


    [Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=4)]: Done  17 tasks      | elapsed:  1.9min
    [Parallel(n_jobs=4)]: Done  90 tasks      | elapsed: 11.8min
    [Parallel(n_jobs=4)]: Done 150 out of 150 | elapsed: 19.4min finished





    RandomizedSearchCV(cv=5, estimator=SVR(), n_iter=30, n_jobs=4,
                       param_distributions={'C': <scipy.stats._distn_infrastructure.rv_frozen object at 0x132d145e0>,
                                            'gamma': <scipy.stats._distn_infrastructure.rv_frozen object at 0x13225eaf0>,
                                            'kernel': ['linear', 'rbf']},
                       return_train_score=True, scoring='neg_mean_squared_error',
                       verbose=4)




```python
%matplotlib inline
from scipy.stats import recipinvgauss, reciprocal
import matplotlib.pyplot as plt

rv = recipinvgauss(mu=1, scale=0.5)
vals = rv.rvs(size=10000)
plt.hist(vals, bins=40, density=True)
plt.hist(np.log(vals), bins=40, density=True)
print(np.quantile(vals, 0.25))
print(np.quantile(vals, 0.75))

rv1 = reciprocal(1, 1000000)
vals = rv1.rvs(size=10000)
# plt.hist(vals, bins=40, density=True)
plt.hist(np.log(vals), bins=40, density=True)
print(np.quantile(vals, 0.25))
print(np.quantile(vals, 0.75))
```

    0.40364256834993917
    1.322189693395722
    28.2468988692541
    31881.1740923203



    
![png](/assets/images/ml-book/chapter2/exercises_9_1.png)
    



```python
print(grid_search.best_params_)
# Since these were the max we probably want to run it with higher values...
print(grid_search.best_estimator_)
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
```

    {'C': 213086.81055914456, 'gamma': 0.43753758012999744, 'kernel': 'rbf'}
    SVR(C=213086.81055914456, gamma=0.43753758012999744)
    86922.13248255629 {'C': 8.366566042717265, 'gamma': 0.570389688549627, 'kernel': 'linear'}
    108940.09733634934 {'C': 112.90281928704358, 'gamma': 0.34622625583652333, 'kernel': 'rbf'}
    77482.89380061901 {'C': 69496.08270814756, 'gamma': 1.4269585049166456, 'kernel': 'linear'}
    88694.80156763176 {'C': 7.3920771548663655, 'gamma': 3.094038787353801, 'kernel': 'linear'}
    77478.50967107895 {'C': 29100.862636014495, 'gamma': 0.34570001769982, 'kernel': 'linear'}
    77480.15495365753 {'C': 166038.1616977072, 'gamma': 0.716976295083469, 'kernel': 'linear'}
    77173.14674835099 {'C': 7754.122689703942, 'gamma': 0.6299597811263543, 'kernel': 'linear'}
    67179.25553893886 {'C': 213086.81055914456, 'gamma': 0.43753758012999744, 'kernel': 'rbf'}
    75000.92839623214 {'C': 431.7411750880907, 'gamma': 1.4726229965871818, 'kernel': 'linear'}
    77776.81326184724 {'C': 23.73956017569095, 'gamma': 1.5622450995843944, 'kernel': 'linear'}
    76001.31467144821 {'C': 974.7978670291761, 'gamma': 0.10198976545558847, 'kernel': 'rbf'}
    120099.11767685601 {'C': 2.001894505635638, 'gamma': 1.6102803109953276, 'kernel': 'rbf'}
    117017.65816463032 {'C': 12.446434888187543, 'gamma': 0.16010633829072501, 'kernel': 'rbf'}
    119539.14394623176 {'C': 11.535484162183055, 'gamma': 0.6614577812082622, 'kernel': 'rbf'}
    118190.95411652334 {'C': 301.83563049289364, 'gamma': 2.075303005602983, 'kernel': 'rbf'}
    71666.87015669054 {'C': 40331.54708973871, 'gamma': 0.6715290369027956, 'kernel': 'rbf'}
    76773.79716060465 {'C': 4707.555909403828, 'gamma': 0.7039968882510544, 'kernel': 'linear'}
    118967.13144966713 {'C': 39.44354980821416, 'gamma': 0.9170816813828603, 'kernel': 'rbf'}
    75946.42891611365 {'C': 848.561483333006, 'gamma': 0.0817326902648654, 'kernel': 'rbf'}
    74421.87334351259 {'C': 180.25902644638163, 'gamma': 0.784200136959768, 'kernel': 'linear'}
    75764.03377391145 {'C': 24574.588703531022, 'gamma': 0.7747648572752451, 'kernel': 'rbf'}
    119997.23391248379 {'C': 1.3971315281598753, 'gamma': 0.47727455561440163, 'kernel': 'rbf'}
    86005.72605371661 {'C': 69044.35646561596, 'gamma': 2.0280319619816884, 'kernel': 'rbf'}
    117951.7838403175 {'C': 143.71272921390602, 'gamma': 1.3280660854622348, 'kernel': 'rbf'}
    117743.1476132982 {'C': 539.3363652863815, 'gamma': 2.462101901209176, 'kernel': 'rbf'}
    68200.58350364857 {'C': 8009.16322915411, 'gamma': 0.16427762321547978, 'kernel': 'rbf'}
    78229.59098213134 {'C': 20405.111727373547, 'gamma': 0.8508539673679466, 'kernel': 'rbf'}
    96531.64243487571 {'C': 4960.87253801678, 'gamma': 1.2603250015966314, 'kernel': 'rbf'}
    69758.88084842438 {'C': 13027.251555935156, 'gamma': 0.3373399035670439, 'kernel': 'rbf'}
    76514.20927753948 {'C': 3268.792643448487, 'gamma': 1.2047119199309158, 'kernel': 'linear'}



```python
# Best was 67179.25553893886 {'C': 213086.81055914456, 'gamma': 0.43753758012999744, 'kernel': 'rbf'}
# Better than before, but it probably make more sense to learn
# about hyperparameters and how they work, than just guessing randomly
```

## Exercise 3
Add a transformer in the pipeline to select only the most important attributes.


```python
from sklearn.base import BaseEstimator, TransformerMixin

feature_importances = [
    (0.3317845401920315, "median_income"),
    (0.14391320344674868, "INLAND"),
    (0.10526089823364354, "population_per_household"),
    (0.08263855622539133, "bedrooms_per_room"),
    (0.08109436950269967, "longitude"),
    (0.06119936528237925, "latitude"),
    (0.05437513667126127, "rooms_per_household"),
    (0.04269180191935387, "housing_median_age"),
    (0.018543650605563098, "population"),
    (0.017855965561009164, "total_rooms"),
    (0.01747459825864214, "total_bedrooms"),
    (0.016371631697584668, "households"),
    (0.015137593949840484, "<1H OCEAN"),
    (0.006837130390816489, "NEAR OCEAN"),
    (0.004801246718794319, "NEAR BAY"),
    (2.0311344240502856e-05, "ISLAND"),
]

encoder_classes = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]


class ImportantFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_importances, cols, features=8):
        self.feature_importances = feature_importances
        self.cols = cols
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cs = [i[1] for i in self.feature_importances[: self.features]]
        # Sort here to maintain order
        idxs = sorted([self.cols.index(i) for i in cs])
        return X[:, idxs]


# Pipeline
cat_cols = ["ocean_proximity"]
num_cols = [col for col in x.columns if col not in cat_cols]

num_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("std_scaler", StandardScaler()),
    ]
)

pipeline = ColumnTransformer(
    [
        ("num", num_pipeline, num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

pipeline = Pipeline(
    [
        ("full_pipeline", pipeline),
        (
            "important_selector",
            ImportantFeatureSelector(
                feature_importances, num_cols + encoder_classes, features=8
            ),
        ),
    ]
)
x_final = pipeline.fit_transform(x)
y_final = y.copy().values[:, 0]
# I double checked that this worked, but commenting out the Standard Scaler
# and comparing to first row of x and x_final
```

## Exercise 4
Try creating a single pipeline that does the full data preparation plus the final prediction.


```python
ex4_pipeline = Pipeline(
    [
        ("prev_pipeline", pipeline),
        ("svr", SVR(**grid_search.best_params_)),
    ]
)
# This runs .transform on prev_pipeline(?), seems like it did
ex4_model = ex4_pipeline.fit(x, y_final)
```


```python
ex4_data = x.iloc[:4]
ex4_labels = y_final[:4]
print(ex4_model.predict(ex4_data).round())
print(ex4_labels)
```

    [448803. 424097. 436690. 313041.]
    [452600. 358500. 352100. 341300.]


## Exercise 5
Automatically explore some preparation options using `GridSearchCV`.


```python
print("\n".join(list(ex4_pipeline.get_params().keys())))
```

    memory
    steps
    verbose
    prev_pipeline
    svr
    prev_pipeline__memory
    prev_pipeline__steps
    prev_pipeline__verbose
    prev_pipeline__full_pipeline
    prev_pipeline__important_selector
    prev_pipeline__full_pipeline__n_jobs
    prev_pipeline__full_pipeline__remainder
    prev_pipeline__full_pipeline__sparse_threshold
    prev_pipeline__full_pipeline__transformer_weights
    prev_pipeline__full_pipeline__transformers
    prev_pipeline__full_pipeline__verbose
    prev_pipeline__full_pipeline__num
    prev_pipeline__full_pipeline__cat
    prev_pipeline__full_pipeline__num__memory
    prev_pipeline__full_pipeline__num__steps
    prev_pipeline__full_pipeline__num__verbose
    prev_pipeline__full_pipeline__num__imputer
    prev_pipeline__full_pipeline__num__std_scaler
    prev_pipeline__full_pipeline__num__imputer__add_indicator
    prev_pipeline__full_pipeline__num__imputer__copy
    prev_pipeline__full_pipeline__num__imputer__fill_value
    prev_pipeline__full_pipeline__num__imputer__missing_values
    prev_pipeline__full_pipeline__num__imputer__strategy
    prev_pipeline__full_pipeline__num__imputer__verbose
    prev_pipeline__full_pipeline__num__std_scaler__copy
    prev_pipeline__full_pipeline__num__std_scaler__with_mean
    prev_pipeline__full_pipeline__num__std_scaler__with_std
    prev_pipeline__full_pipeline__cat__categories
    prev_pipeline__full_pipeline__cat__drop
    prev_pipeline__full_pipeline__cat__dtype
    prev_pipeline__full_pipeline__cat__handle_unknown
    prev_pipeline__full_pipeline__cat__sparse
    prev_pipeline__important_selector__cols
    prev_pipeline__important_selector__feature_importances
    prev_pipeline__important_selector__features
    svr__C
    svr__cache_size
    svr__coef0
    svr__degree
    svr__epsilon
    svr__gamma
    svr__kernel
    svr__max_iter
    svr__shrinking
    svr__tol
    svr__verbose



```python
grid_params = [
    {
        "prev_pipeline__full_pipeline__num__imputer__strategy": [
            "mean",
            "most_frequent",
        ],
        "prev_pipeline__important_selector__features": list(range(3, 6)),
        "svr__epsilon": np.logspace(-2, 0, 3),
    }
]

gs = GridSearchCV(
    ex4_pipeline,
    grid_params,
    cv=5,
    scoring="neg_mean_squared_error",
    return_train_score=True,
    n_jobs=4,
    verbose=4,
)
gs.fit(x, y_final)
```

    Fitting 5 folds for each of 18 candidates, totalling 90 fits


    [Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=4)]: Done  17 tasks      | elapsed:  1.4min
    [Parallel(n_jobs=4)]: Done  90 out of  90 | elapsed:  6.9min finished





    GridSearchCV(cv=5,
                 estimator=Pipeline(steps=[('prev_pipeline',
                                            Pipeline(steps=[('full_pipeline',
                                                             ColumnTransformer(transformers=[('num',
                                                                                              Pipeline(steps=[('imputer',
                                                                                                               SimpleImputer(strategy='median')),
                                                                                                              ('std_scaler',
                                                                                                               StandardScaler())]),
                                                                                              ['longitude',
                                                                                               'latitude',
                                                                                               'housing_median_age',
                                                                                               'total_rooms',
                                                                                               'total_bedrooms',
                                                                                               'population',
                                                                                               'households',
                                                                                               'median_income',
                                                                                               'pop...
                                                                                                           (2.0311344240502856e-05,
                                                                                                            'ISLAND')]))])),
                                           ('svr',
                                            SVR(C=213086.81055914456,
                                                gamma=0.43753758012999744))]),
                 n_jobs=4,
                 param_grid=[{'prev_pipeline__full_pipeline__num__imputer__strategy': ['mean',
                                                                                       'most_frequent'],
                              'prev_pipeline__important_selector__features': [3, 4,
                                                                              5],
                              'svr__epsilon': array([0.01, 0.1 , 1.  ])}],
                 return_train_score=True, scoring='neg_mean_squared_error',
                 verbose=4)




```python
print(gs.best_params_)
# Since these were the max we probably want to run it with higher values...
print(gs.best_estimator_)
cvres = gs.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
```

    {'prev_pipeline__full_pipeline__num__imputer__strategy': 'most_frequent', 'prev_pipeline__important_selector__features': 4, 'svr__epsilon': 0.01}
    Pipeline(steps=[('prev_pipeline',
                     Pipeline(steps=[('full_pipeline',
                                      ColumnTransformer(transformers=[('num',
                                                                       Pipeline(steps=[('imputer',
                                                                                        SimpleImputer(strategy='most_frequent')),
                                                                                       ('std_scaler',
                                                                                        StandardScaler())]),
                                                                       ['longitude',
                                                                        'latitude',
                                                                        'housing_median_age',
                                                                        'total_rooms',
                                                                        'total_bedrooms',
                                                                        'population',
                                                                        'households',
                                                                        'median_income',
                                                                        'population_per_household...
                                                                                     'population'),
                                                                                    (0.017855965561009164,
                                                                                     'total_rooms'),
                                                                                    (0.01747459825864214,
                                                                                     'total_bedrooms'),
                                                                                    (0.016371631697584668,
                                                                                     'households'),
                                                                                    (0.015137593949840484,
                                                                                     '<1H '
                                                                                     'OCEAN'),
                                                                                    (0.006837130390816489,
                                                                                     'NEAR '
                                                                                     'OCEAN'),
                                                                                    (0.004801246718794319,
                                                                                     'NEAR '
                                                                                     'BAY'),
                                                                                    (2.0311344240502856e-05,
                                                                                     'ISLAND')],
                                                               features=4))])),
                    ('svr',
                     SVR(C=213086.81055914456, epsilon=0.01,
                         gamma=0.43753758012999744))])
    68518.29884263125 {'prev_pipeline__full_pipeline__num__imputer__strategy': 'mean', 'prev_pipeline__important_selector__features': 3, 'svr__epsilon': 0.01}
    68518.29432576336 {'prev_pipeline__full_pipeline__num__imputer__strategy': 'mean', 'prev_pipeline__important_selector__features': 3, 'svr__epsilon': 0.1}
    68518.23408166702 {'prev_pipeline__full_pipeline__num__imputer__strategy': 'mean', 'prev_pipeline__important_selector__features': 3, 'svr__epsilon': 1.0}
    67035.46282230441 {'prev_pipeline__full_pipeline__num__imputer__strategy': 'mean', 'prev_pipeline__important_selector__features': 4, 'svr__epsilon': 0.01}
    67035.46222793893 {'prev_pipeline__full_pipeline__num__imputer__strategy': 'mean', 'prev_pipeline__important_selector__features': 4, 'svr__epsilon': 0.1}
    67035.4446865113 {'prev_pipeline__full_pipeline__num__imputer__strategy': 'mean', 'prev_pipeline__important_selector__features': 4, 'svr__epsilon': 1.0}
    69814.47717260549 {'prev_pipeline__full_pipeline__num__imputer__strategy': 'mean', 'prev_pipeline__important_selector__features': 5, 'svr__epsilon': 0.01}
    69814.47002786033 {'prev_pipeline__full_pipeline__num__imputer__strategy': 'mean', 'prev_pipeline__important_selector__features': 5, 'svr__epsilon': 0.1}
    69814.4032625619 {'prev_pipeline__full_pipeline__num__imputer__strategy': 'mean', 'prev_pipeline__important_selector__features': 5, 'svr__epsilon': 1.0}
    68518.29884263125 {'prev_pipeline__full_pipeline__num__imputer__strategy': 'most_frequent', 'prev_pipeline__important_selector__features': 3, 'svr__epsilon': 0.01}
    68518.29432576336 {'prev_pipeline__full_pipeline__num__imputer__strategy': 'most_frequent', 'prev_pipeline__important_selector__features': 3, 'svr__epsilon': 0.1}
    68518.23408166702 {'prev_pipeline__full_pipeline__num__imputer__strategy': 'most_frequent', 'prev_pipeline__important_selector__features': 3, 'svr__epsilon': 1.0}
    66997.77217958865 {'prev_pipeline__full_pipeline__num__imputer__strategy': 'most_frequent', 'prev_pipeline__important_selector__features': 4, 'svr__epsilon': 0.01}
    66997.7728305426 {'prev_pipeline__full_pipeline__num__imputer__strategy': 'most_frequent', 'prev_pipeline__important_selector__features': 4, 'svr__epsilon': 0.1}
    66997.79453259113 {'prev_pipeline__full_pipeline__num__imputer__strategy': 'most_frequent', 'prev_pipeline__important_selector__features': 4, 'svr__epsilon': 1.0}
    69799.97323191866 {'prev_pipeline__full_pipeline__num__imputer__strategy': 'most_frequent', 'prev_pipeline__important_selector__features': 5, 'svr__epsilon': 0.01}
    69799.96662789413 {'prev_pipeline__full_pipeline__num__imputer__strategy': 'most_frequent', 'prev_pipeline__important_selector__features': 5, 'svr__epsilon': 0.1}
    69799.9048895546 {'prev_pipeline__full_pipeline__num__imputer__strategy': 'most_frequent', 'prev_pipeline__important_selector__features': 5, 'svr__epsilon': 1.0}


* It's super convenient that you can set parameters like that (i.e. using the string version)
* Exercises took too long... SVR takes a long time to fit with cv
