---
layout: notes
chapter: 2
chapter-title: End-to-End Machine Learning Project
permalink: /ml-book/chapter2/notes.html
---

## The Process
* Big Picture (Problem Statement)
* Get Data
* Exploratory Data Analysis
* Data Preparation
* Model selection and training
* Fine-tune the model
* Production
* Monitor and Maintain

## Frame the Problem

### The Task
* Build a model to predict housing prices in California given California census data. Specifically, predict the median housing price in any district, given all other metrics. 

### Additional Considerations
* Determine how exactly the result of your model is going to be used
  * In this instance it will be fed into another machine learning model downstream
  * Current process is a manual one which is costly and time consuming
  * Typical error rate of the experts is 15%
* Questions to ask yourself:
  * Is it supervised, unsupervised, or Reinforcement Learning? Supervised (because we have labels of existing median housing price
  * Is it a classification, regression or something else? It's a regression task, we're predicting a number
  * Should you use batch learning or online learning? Depends on the volume of data, but probably batch learning.

## RMSE (Root Mean Squared Error)
Measures the standard deviation of the errors the system makes in its predictions. Recall the standard deviation is:

$$ \sigma = \sqrt{\frac{\sum_{i}{(\bar{X} - X_i)^2}}{N}} $$

Analogously RMSE is:

$$ RMSE = \sqrt{\frac{\sum_{i}{(y_i - f(X_i))^2}}{N}} $$

where $f$ is our model. There is also Mean Absolute Error (MAE). RMSE is more 
sensitive to outliers than MAE because for large outliers (i.e. differences) 
RMSE will make them larger by squaring them.

## Get the Data


```python
import tarfile
import tempfile
import urllib.request
import os
import pandas as pd

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
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-122.23</td>
      <td>37.88</td>
      <td>41.0</td>
      <td>880.0</td>
      <td>129.0</td>
      <td>322.0</td>
      <td>126.0</td>
      <td>8.3252</td>
      <td>452600.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-122.22</td>
      <td>37.86</td>
      <td>21.0</td>
      <td>7099.0</td>
      <td>1106.0</td>
      <td>2401.0</td>
      <td>1138.0</td>
      <td>8.3014</td>
      <td>358500.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-122.24</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>1467.0</td>
      <td>190.0</td>
      <td>496.0</td>
      <td>177.0</td>
      <td>7.2574</td>
      <td>352100.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-122.25</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>1274.0</td>
      <td>235.0</td>
      <td>558.0</td>
      <td>219.0</td>
      <td>5.6431</td>
      <td>341300.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-122.25</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>1627.0</td>
      <td>280.0</td>
      <td>565.0</td>
      <td>259.0</td>
      <td>3.8462</td>
      <td>342200.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>20635</th>
      <td>-121.09</td>
      <td>39.48</td>
      <td>25.0</td>
      <td>1665.0</td>
      <td>374.0</td>
      <td>845.0</td>
      <td>330.0</td>
      <td>1.5603</td>
      <td>78100.0</td>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>20636</th>
      <td>-121.21</td>
      <td>39.49</td>
      <td>18.0</td>
      <td>697.0</td>
      <td>150.0</td>
      <td>356.0</td>
      <td>114.0</td>
      <td>2.5568</td>
      <td>77100.0</td>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>20637</th>
      <td>-121.22</td>
      <td>39.43</td>
      <td>17.0</td>
      <td>2254.0</td>
      <td>485.0</td>
      <td>1007.0</td>
      <td>433.0</td>
      <td>1.7000</td>
      <td>92300.0</td>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>20638</th>
      <td>-121.32</td>
      <td>39.43</td>
      <td>18.0</td>
      <td>1860.0</td>
      <td>409.0</td>
      <td>741.0</td>
      <td>349.0</td>
      <td>1.8672</td>
      <td>84700.0</td>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>20639</th>
      <td>-121.24</td>
      <td>39.37</td>
      <td>16.0</td>
      <td>2785.0</td>
      <td>616.0</td>
      <td>1387.0</td>
      <td>530.0</td>
      <td>2.3886</td>
      <td>89400.0</td>
      <td>INLAND</td>
    </tr>
  </tbody>
</table>
<p>20640 rows × 10 columns</p>
</div>




```python
# Show histogram of the features
%matplotlib inline
import matplotlib.pyplot as plt

df.hist(bins=100, figsize=(16, 12))
```


    
![png](/assets/images/ml-book/chapter2/notes_6_0.png)
    



```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20433.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-119.569704</td>
      <td>35.631861</td>
      <td>28.639486</td>
      <td>2635.763081</td>
      <td>537.870553</td>
      <td>1425.476744</td>
      <td>499.539680</td>
      <td>3.870671</td>
      <td>206855.816909</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.003532</td>
      <td>2.135952</td>
      <td>12.585558</td>
      <td>2181.615252</td>
      <td>421.385070</td>
      <td>1132.462122</td>
      <td>382.329753</td>
      <td>1.899822</td>
      <td>115395.615874</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-124.350000</td>
      <td>32.540000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>0.499900</td>
      <td>14999.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-121.800000</td>
      <td>33.930000</td>
      <td>18.000000</td>
      <td>1447.750000</td>
      <td>296.000000</td>
      <td>787.000000</td>
      <td>280.000000</td>
      <td>2.563400</td>
      <td>119600.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-118.490000</td>
      <td>34.260000</td>
      <td>29.000000</td>
      <td>2127.000000</td>
      <td>435.000000</td>
      <td>1166.000000</td>
      <td>409.000000</td>
      <td>3.534800</td>
      <td>179700.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>-118.010000</td>
      <td>37.710000</td>
      <td>37.000000</td>
      <td>3148.000000</td>
      <td>647.000000</td>
      <td>1725.000000</td>
      <td>605.000000</td>
      <td>4.743250</td>
      <td>264725.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>-114.310000</td>
      <td>41.950000</td>
      <td>52.000000</td>
      <td>39320.000000</td>
      <td>6445.000000</td>
      <td>35682.000000</td>
      <td>6082.000000</td>
      <td>15.000100</td>
      <td>500001.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Things to note
* Several columns seem to be capped out a max value (e.g. `housing_median_age`)
* `median_income` isn't in dollars
* Many distributions are right-skewed (hump on left, called tail-heavy)

### Things mentioned by Geron
* `median_income` isn't in dollars
* `housing_median_age` and `median_housing_value` are capped
  * The latter might be problematic because it is our target variable. To remedy he suggests:
    * Collect correct labels for those
    * Remove those districts from the training/test set
* Different scales
* Tail heavy

## Create a Test Set
* Most of the time you're going to be fine with randomly sampling/splitting into train/test
* Geron suggests stratified sampling (5 splits) based on median income
* We'll try both with a 20% test size


```python
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
```

Let's see how similar these are:


```python
ax = train_set.hist(bins=5, figsize=(16, 12), density=True, alpha=0.8)
test_set.hist(bins=5, figsize=(16, 12), ax=ax, density=True, alpha=0.8)
```


    
![png](/assets/images/ml-book/chapter2/notes_12_0.png)
    


That looks pretty good to me. But we'll also do the stratified method:


```python
# Sample to same count in each bin (5 bins)
# We can come back and try this at the end to see how if the performance improves?
import numpy as np
import pandas as pd

strat_values = df["median_income"]
bins = 5
x = np.linspace(0, len(strat_values), bins + 1)
xp = np.arange(len(strat_values))
fp = np.sort(strat_values)
bin_ends = np.interp(x, xp, fp)
# Make sure we include the bin ends and end up with 5 bins in the end
bin_ends[0] -= 0.001
bin_ends[-1] += 0.001
strat = np.digitize(strat_values, bins=bin_ends, right=True)
print(bin_ends)
print(pd.value_counts(strat))
df["income_cat"] = strat
strat_train_set, strat_test_set = train_test_split(
    df, test_size=0.2, random_state=42, stratify=strat
)
ax = strat_train_set.hist(bins=5, figsize=(16, 16), density=True, alpha=0.8)
strat_test_set.hist(
    bins=5, figsize=(16, 16), ax=ax.flatten()[:-2], density=True, alpha=0.8
)
```

    [ 0.4989  2.3523  3.1406  3.9673  5.1098 15.0011]
    2    4131
    1    4130
    4    4128
    5    4127
    3    4124
    dtype: int64



    
![png](/assets/images/ml-book/chapter2/notes_14_1.png)
    



```python
strat_values = df["median_income"]
bins = 5
strat = np.ceil(strat_values / 1.5)
strat = strat.where(strat < 5, 5.0)
df["income_cat"] = strat
print(pd.value_counts(strat) / len(strat))
strat_train_set, strat_test_set = train_test_split(
    df, test_size=0.2, random_state=42, stratify=strat
)
ax = strat_train_set.hist(bins=5, figsize=(16, 16), density=True, alpha=0.8)
strat_test_set.hist(
    bins=5, figsize=(16, 16), ax=ax.flatten()[:-2], density=True, alpha=0.8
)
```

    3.0    0.350581
    2.0    0.318847
    4.0    0.176308
    5.0    0.114438
    1.0    0.039826
    Name: median_income, dtype: float64



    
![png](/assets/images/ml-book/chapter2/notes_15_1.png)
    


I feel like this doesn't matter at all...


```python
# Drop the income_cat column
cols = [i for i in df.columns if i != "income_cat"]
df = df.loc[:, cols]
strat_train_set = strat_train_set.loc[:, cols]
strat_test_set = strat_test_set.loc[:, cols]
# Only work with train set from here on out
df = strat_train_set.copy()
```

## Visualize the Data to Gain Insights
* Visualize geographically based on target variable
* Correlations
* Combining features


```python
import seaborn as sns

plt.figure(figsize=(12, 12))
sns.scatterplot(
    x="longitude",
    y="latitude",
    data=df,
    s=df["population"] / 50,
    hue=df["median_house_value"],
    alpha=0.3,
    palette="seismic",
)
plt.title("Geographical Population/House Value Plot")
```


    
![png](/assets/images/ml-book/chapter2/notes_19_0.png)
    



```python
# Correlations
corr = df.corr()
corr["median_house_value"].sort_values(ascending=False)
# Scatter
from pandas.plotting import scatter_matrix

scatter_matrix(
    df[["median_house_value", "median_income", "total_rooms", "housing_median_age"]],
    figsize=(16, 12),
)
```


    
![png](/assets/images/ml-book/chapter2/notes_20_0.png)
    



```python
# Combining features
df["population_per_household"] = df["population"] / df["households"]
df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
df["rooms_per_household"] = df["total_rooms"] / df["households"]
df.corr()["median_house_value"].sort_values(ascending=False)
# Not sure why rooms_per_household was 0.05 less than Geron...
```




    median_house_value          1.000000
    median_income               0.687160
    rooms_per_household         0.146285
    total_rooms                 0.135097
    housing_median_age          0.114110
    households                  0.064506
    total_bedrooms              0.047689
    population_per_household   -0.021985
    population                 -0.026920
    longitude                  -0.047432
    latitude                   -0.142724
    bedrooms_per_room          -0.259984
    Name: median_house_value, dtype: float64



## Prepare the Data for Machine Learning Algorithms
* Data Cleaning
  * Handle missing data in `total_bedrooms`
    * Option 1: Remove column entirely (kind of a lousy option considering only a few districts are missing and we just created a combo feature based on it)
    * Option 2: Remove those districts (have to remove from test set as well)...
    * Option 3: Fill value (mean, median, etc.)
  * We'll just go with option 3, using the median as Geron does
  * He makes a good point, that we should use the imputer on _all_ numerical variables because for future data there might be missing data in other columns


```python
target = "median_house_value"
x = df[[col for col in df.columns if col != target]].copy()
y = df[[target]].copy()
```


```python
# Impute the median
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
imputer.fit(x.drop(columns="ocean_proximity"))
print(list(imputer.statistics_.round(2)))
x_num = imputer.transform(x.drop(columns="ocean_proximity"))
print(x_num)
```

    [-118.51, 34.26, 29.0, 2119.5, 433.0, 1164.0, 408.0, 3.54, 2.82, 0.2, 5.23]
    [[-121.89         37.29         38.         ...    2.09439528
         0.22385204    4.62536873]
     [-121.93         37.05         14.         ...    2.7079646
         0.15905744    6.00884956]
     [-117.2          32.77         31.         ...    2.02597403
         0.24129098    4.22510823]
     ...
     [-116.4          34.09          9.         ...    2.74248366
         0.17960865    6.34640523]
     [-118.01         33.82         31.         ...    3.80898876
         0.19387755    5.50561798]
     [-122.45         37.77         52.         ...    1.98591549
         0.22035541    4.84350548]]


## Scikit-Learn Design
* Consistency
  * All objects share a consistent and simple interface:
    * Esimators: 
      * Any object can estimate some parameters based on a dataset
      * Estimation is performed by calling `fit`
      * Hyperparameters are set at instantiation
    * Transformers:
      * Estimators which can transform a dataset
      * Transformation is performed by calling `transform` with the dataset as the arg
      * It returns the transformed dataset
      * Some transformers have an optimized `fit_transform` method which runs both steps
    * Predictors:
      * Estimators which can make predictions on a dataset
      * Prediction is performed by calling `predict` with the new dataset as the arg
      * They also have a score used to evaluate the quality of predictions
  * Inspection:
    * Hyperparameters of estimators are available in public instance variables
    * Learn parameters of estimators are available, and their variables end with an underscore
  * Nonproliferation of classes:
    * Datasets are numpy or scipy arrays or sparse matrices
    * Hyperparameters are python datatypes
  * Composition:
    * Existing building blocks are resused as much as possible
  * Sensible defaults:
    * Estimators have sensible defaults for their hyperparameters
  

## Handling Text and Categorical Attributes
* Machine learning algorithms need to work with numbers so we encode textual data as numerical input
* A label encoder will map labels into integers
  * But, most ML algorithms will assume that numbers closer together are more similar
* Therefore we can use a one hot encoding to create binary labels for each category
* `LabelBinarizer` is the combination of these steps


```python
from sklearn.preprocessing import LabelBinarizer

encoder = LabelBinarizer()
x_cat = encoder.fit_transform(x[["ocean_proximity"]])
x_cat
# OneHotEncoder functionality has improved so we use that later on in favor of LabelBinarizer
```




    array([[1, 0, 0, 0, 0],
           [1, 0, 0, 0, 0],
           [0, 0, 0, 0, 1],
           ...,
           [0, 1, 0, 0, 0],
           [1, 0, 0, 0, 0],
           [0, 0, 0, 1, 0]])



I'm going to avoid the custom transformer for now.

## Feature Scaling
* **With few exceptions, ML algorithms do not perform well when the input numerical attributes have very different scales**
* Min-max scaling or Normalization: values are rescaled to 0 - 1
  * Use `MinMaxScaler`
* Standardization: Zero mean and unit variance
  * Much less affected by outliers
  * Use `StandardScaler`
* Only feature scale on training set


```python
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

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
print(x_final)
print(x_final.shape)
# In the copy of the book I have the shape is (16513, 17),
# but in the updated version
# online: https://github.com/ageron/handson-ml/blob
# /master/02_end_to_end_machine_learning_project.ipynb
# it is (16512, 16)
```

    [[-1.15604281  0.77194962  0.74333089 ...  0.          0.
       0.        ]
     [-1.17602483  0.6596948  -1.1653172  ...  0.          0.
       0.        ]
     [ 1.18684903 -1.34218285  0.18664186 ...  0.          0.
       1.        ]
     ...
     [ 1.58648943 -0.72478134 -1.56295222 ...  0.          0.
       0.        ]
     [ 0.78221312 -0.85106801  0.18664186 ...  0.          0.
       0.        ]
     [-1.43579109  0.99645926  1.85670895 ...  0.          1.
       0.        ]]
    (16512, 16)


## A Note
It's good to reference [the notebooks here](https://github.com/ageron/handson-ml) because Geron updated them with new ideas and changes that have been made in new scikit-learn versions! Example of this above is `ColumnTransformer` and the change in behavior of `OneHotEncoder`.

## Select and Train a Model

   >At last! You framed the problem, you got the data and explored it, you sampled a training set and a test set, and you wrote transformation pipelines to clean up and prepare your data for Machine Learning algorithms automatically. You are now ready to select and train a Machine Learning model.
   
That was straight from Geron, the excitement is palpable :)

## Training and Evaluating on the Training Set


```python
from sklearn.linear_model import LinearRegression

y_final = y.copy().values

lin_reg = LinearRegression()
lin_reg.fit(x_final, y_final)

# Some predictions
x_5 = x_final[:5, :]
y_5 = y_final[:5]
print(list(lin_reg.predict(x_5)[:, 0].round(2)))
print(list(y_5[:, 0]))
```

    [209375.74, 315154.78, 210238.28, 55902.62, 183416.69]
    [286600.0, 340600.0, 196900.0, 46300.0, 254500.0]



```python
from sklearn.metrics import mean_squared_error

lin_preds = lin_reg.predict(x_final)
lin_mse = mean_squared_error(y_final, lin_preds)
lin_rmse = np.sqrt(lin_mse)
lin_rmse  # Better than Geron :)
```




    68161.22644433199




```python
# Let's plot this
plt.figure(figsize=FIGSIZE)
plt.scatter(lin_preds[:, 0], y_final[:, 0])
plt.plot(np.arange(max(y_final[:, 0])), np.arange(max(y_final[:, 0])), c="r", lw=4)
plt.axis("equal")
plt.xlabel("Predictions")
plt.ylabel("Labels")
plt.title("Median House Price: Predictions vs. Labels")
```




    Text(0.5, 1.0, 'Median House Price: Predictions vs. Labels')




    
![png](/assets/images/ml-book/chapter2/notes_36_1.png)
    



```python
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(x_final, y_final)
tree_preds = tree_reg.predict(x_final)
tree_mse = mean_squared_error(y_final, tree_preds)
tree_rmse = np.sqrt(tree_mse)
tree_rmse
```




    0.0



## Underfitting and Overfitting
* Clearly the LinearRegression model is underfitting
* The DecisionTreeRegressor model is overfitting

## K-fold Cross Validation
* Split the training data k folds
* Iterate it k times using k-1 folds for training and one for the test set
* Evaluate on the test_set k times


```python
from sklearn.model_selection import cross_val_score


def cross_val_model(m, x_m, y_m, cv=10):
    scores = cross_val_score(m, x_m, y_m, scoring="neg_mean_squared_error", cv=cv)
    rmse_scores = np.sqrt(-scores)
    print(rmse_scores, np.mean(rmse_scores), np.std(rmse_scores))


cross_val_model(tree_reg, x_final, y_final)
```

    [70968.72056379 67216.68718226 70857.65880397 69194.86108641
     69756.29757786 74386.85421573 69949.56290335 69745.34537599
     75022.85006194 70755.92128417] 70785.47590554852 2215.1990085744



```python
cross_val_model(lin_reg, x_final, y_final)
```

    [66060.65470195 66764.30726969 67721.72734022 74719.28193624
     68058.11572078 70909.35812986 64171.66459204 68075.65317717
     71024.84033989 67300.24394751] 68480.58471553595 2845.5843092650853



```python
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor()
forest_reg.fit(x_final, y_final[:, 0])
forest_preds = forest_reg.predict(x_final)
forest_mse = mean_squared_error(forest_preds, y_final)
forest_rmse = np.sqrt(forest_mse)
print(forest_rmse)
cross_val_model(forest_reg, x_final, y_final[:, 0])
```

    18681.372911866638
    [49635.15372436 47754.83871792 49368.25902706 51887.71850715
     49747.11331684 53513.21033152 49044.38099493 47851.45135021
     52535.51927089 50181.50476447] 50151.91500053619 1824.9254115323



```python
plt.figure(figsize=FIGSIZE)
plt.scatter(forest_preds, y_final[:, 0])
plt.plot(np.arange(max(y_final[:, 0])), np.arange(max(y_final[:, 0])), c="r", lw=4)
plt.axis("equal")
plt.xlabel("Predictions")
plt.ylabel("Labels")
plt.title("Median House Price (forest_reg): Predictions vs. Labels")
```




    Text(0.5, 1.0, 'Median House Price (forest_reg): Predictions vs. Labels')




    
![png](/assets/images/ml-book/chapter2/notes_42_1.png)
    


## Fine-Tune Your Model
* Hyper parameter tuning via `GridSearchCV`


```python
from sklearn.model_selection import GridSearchCV

param_grid = [
    {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
    {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(
    forest_reg,
    param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    return_train_score=True,
)
grid_search.fit(x_final, y_final[:, 0])
```




    GridSearchCV(cv=5, estimator=RandomForestRegressor(),
                 param_grid=[{'max_features': [2, 4, 6, 8],
                              'n_estimators': [3, 10, 30]},
                             {'bootstrap': [False], 'max_features': [2, 3, 4],
                              'n_estimators': [3, 10]}],
                 return_train_score=True, scoring='neg_mean_squared_error')




```python
print(grid_search.best_params_)
# Since these were the max we probably want to run it with higher values...
print(grid_search.best_estimator_)
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
```

    {'max_features': 6, 'n_estimators': 30}
    RandomForestRegressor(max_features=6, n_estimators=30)
    64241.643733474426 {'max_features': 2, 'n_estimators': 3}
    55493.95031384657 {'max_features': 2, 'n_estimators': 10}
    52611.35103475831 {'max_features': 2, 'n_estimators': 30}
    59612.070906720124 {'max_features': 4, 'n_estimators': 3}
    53217.142164320154 {'max_features': 4, 'n_estimators': 10}
    50774.31443657333 {'max_features': 4, 'n_estimators': 30}
    58496.50322845977 {'max_features': 6, 'n_estimators': 3}
    51673.17455491991 {'max_features': 6, 'n_estimators': 10}
    49905.43427800321 {'max_features': 6, 'n_estimators': 30}
    58415.20435335512 {'max_features': 8, 'n_estimators': 3}
    51769.879435332965 {'max_features': 8, 'n_estimators': 10}
    50108.24515443716 {'max_features': 8, 'n_estimators': 30}
    63641.17748807948 {'bootstrap': False, 'max_features': 2, 'n_estimators': 3}
    54078.8506451545 {'bootstrap': False, 'max_features': 2, 'n_estimators': 10}
    60653.00976167665 {'bootstrap': False, 'max_features': 3, 'n_estimators': 3}
    52864.94701183964 {'bootstrap': False, 'max_features': 3, 'n_estimators': 10}
    57904.55694831166 {'bootstrap': False, 'max_features': 4, 'n_estimators': 3}
    51990.81114906108 {'bootstrap': False, 'max_features': 4, 'n_estimators': 10}



```python
# 49744.32698468949 is better than the 50063.56307010515 that we got earlier
pd.DataFrame(grid_search.cv_results_)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean_fit_time</th>
      <th>std_fit_time</th>
      <th>mean_score_time</th>
      <th>std_score_time</th>
      <th>param_max_features</th>
      <th>param_n_estimators</th>
      <th>param_bootstrap</th>
      <th>params</th>
      <th>split0_test_score</th>
      <th>split1_test_score</th>
      <th>...</th>
      <th>mean_test_score</th>
      <th>std_test_score</th>
      <th>rank_test_score</th>
      <th>split0_train_score</th>
      <th>split1_train_score</th>
      <th>split2_train_score</th>
      <th>split3_train_score</th>
      <th>split4_train_score</th>
      <th>mean_train_score</th>
      <th>std_train_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.074970</td>
      <td>0.003440</td>
      <td>0.004058</td>
      <td>0.000327</td>
      <td>2</td>
      <td>3</td>
      <td>NaN</td>
      <td>{'max_features': 2, 'n_estimators': 3}</td>
      <td>-3.725274e+09</td>
      <td>-4.519071e+09</td>
      <td>...</td>
      <td>-4.126989e+09</td>
      <td>2.782979e+08</td>
      <td>18</td>
      <td>-1.174094e+09</td>
      <td>-1.137123e+09</td>
      <td>-1.135578e+09</td>
      <td>-1.161493e+09</td>
      <td>-1.173606e+09</td>
      <td>-1.156379e+09</td>
      <td>1.697186e+07</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.253067</td>
      <td>0.006757</td>
      <td>0.013049</td>
      <td>0.000900</td>
      <td>2</td>
      <td>10</td>
      <td>NaN</td>
      <td>{'max_features': 2, 'n_estimators': 10}</td>
      <td>-2.902669e+09</td>
      <td>-3.157306e+09</td>
      <td>...</td>
      <td>-3.079579e+09</td>
      <td>1.487250e+08</td>
      <td>11</td>
      <td>-5.985522e+08</td>
      <td>-5.762183e+08</td>
      <td>-5.598208e+08</td>
      <td>-5.842322e+08</td>
      <td>-5.538692e+08</td>
      <td>-5.745385e+08</td>
      <td>1.623130e+07</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.766241</td>
      <td>0.061197</td>
      <td>0.041019</td>
      <td>0.012172</td>
      <td>2</td>
      <td>30</td>
      <td>NaN</td>
      <td>{'max_features': 2, 'n_estimators': 30}</td>
      <td>-2.522423e+09</td>
      <td>-2.879564e+09</td>
      <td>...</td>
      <td>-2.767954e+09</td>
      <td>1.561066e+08</td>
      <td>7</td>
      <td>-4.315321e+08</td>
      <td>-4.233639e+08</td>
      <td>-4.226476e+08</td>
      <td>-4.443902e+08</td>
      <td>-4.293808e+08</td>
      <td>-4.302629e+08</td>
      <td>7.842951e+06</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.117724</td>
      <td>0.002772</td>
      <td>0.004679</td>
      <td>0.000370</td>
      <td>4</td>
      <td>3</td>
      <td>NaN</td>
      <td>{'max_features': 4, 'n_estimators': 3}</td>
      <td>-3.327938e+09</td>
      <td>-3.817583e+09</td>
      <td>...</td>
      <td>-3.553599e+09</td>
      <td>2.235334e+08</td>
      <td>15</td>
      <td>-9.922077e+08</td>
      <td>-9.869361e+08</td>
      <td>-9.914129e+08</td>
      <td>-9.018709e+08</td>
      <td>-9.423203e+08</td>
      <td>-9.629496e+08</td>
      <td>3.577073e+07</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.396204</td>
      <td>0.010106</td>
      <td>0.012332</td>
      <td>0.000672</td>
      <td>4</td>
      <td>10</td>
      <td>NaN</td>
      <td>{'max_features': 4, 'n_estimators': 10}</td>
      <td>-2.740772e+09</td>
      <td>-2.832407e+09</td>
      <td>...</td>
      <td>-2.832064e+09</td>
      <td>1.248181e+08</td>
      <td>9</td>
      <td>-5.299414e+08</td>
      <td>-5.012337e+08</td>
      <td>-5.068510e+08</td>
      <td>-5.150394e+08</td>
      <td>-5.214479e+08</td>
      <td>-5.149027e+08</td>
      <td>1.020482e+07</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.183900</td>
      <td>0.032037</td>
      <td>0.034720</td>
      <td>0.003109</td>
      <td>4</td>
      <td>30</td>
      <td>NaN</td>
      <td>{'max_features': 4, 'n_estimators': 30}</td>
      <td>-2.458769e+09</td>
      <td>-2.659611e+09</td>
      <td>...</td>
      <td>-2.578031e+09</td>
      <td>1.205170e+08</td>
      <td>3</td>
      <td>-3.960356e+08</td>
      <td>-3.967138e+08</td>
      <td>-3.976908e+08</td>
      <td>-3.966141e+08</td>
      <td>-3.885729e+08</td>
      <td>-3.951254e+08</td>
      <td>3.319175e+06</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.163407</td>
      <td>0.006830</td>
      <td>0.004493</td>
      <td>0.000744</td>
      <td>6</td>
      <td>3</td>
      <td>NaN</td>
      <td>{'max_features': 6, 'n_estimators': 3}</td>
      <td>-3.348268e+09</td>
      <td>-3.590983e+09</td>
      <td>...</td>
      <td>-3.421841e+09</td>
      <td>1.263433e+08</td>
      <td>14</td>
      <td>-9.210685e+08</td>
      <td>-9.293810e+08</td>
      <td>-9.024612e+08</td>
      <td>-9.630794e+08</td>
      <td>-8.755641e+08</td>
      <td>-9.183108e+08</td>
      <td>2.902711e+07</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.540489</td>
      <td>0.024100</td>
      <td>0.012174</td>
      <td>0.000949</td>
      <td>6</td>
      <td>10</td>
      <td>NaN</td>
      <td>{'max_features': 6, 'n_estimators': 10}</td>
      <td>-2.539299e+09</td>
      <td>-2.724873e+09</td>
      <td>...</td>
      <td>-2.670117e+09</td>
      <td>1.431570e+08</td>
      <td>4</td>
      <td>-5.289777e+08</td>
      <td>-4.838298e+08</td>
      <td>-4.794369e+08</td>
      <td>-4.998217e+08</td>
      <td>-5.077449e+08</td>
      <td>-4.999622e+08</td>
      <td>1.779906e+07</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.557723</td>
      <td>0.008208</td>
      <td>0.032147</td>
      <td>0.001470</td>
      <td>6</td>
      <td>30</td>
      <td>NaN</td>
      <td>{'max_features': 6, 'n_estimators': 30}</td>
      <td>-2.311978e+09</td>
      <td>-2.530306e+09</td>
      <td>...</td>
      <td>-2.490552e+09</td>
      <td>1.364425e+08</td>
      <td>1</td>
      <td>-3.738081e+08</td>
      <td>-3.723503e+08</td>
      <td>-3.776112e+08</td>
      <td>-3.963427e+08</td>
      <td>-3.891545e+08</td>
      <td>-3.818534e+08</td>
      <td>9.341090e+06</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.199586</td>
      <td>0.008718</td>
      <td>0.003961</td>
      <td>0.000096</td>
      <td>8</td>
      <td>3</td>
      <td>NaN</td>
      <td>{'max_features': 8, 'n_estimators': 3}</td>
      <td>-3.293286e+09</td>
      <td>-3.533084e+09</td>
      <td>...</td>
      <td>-3.412336e+09</td>
      <td>1.883880e+08</td>
      <td>13</td>
      <td>-9.155409e+08</td>
      <td>-8.965800e+08</td>
      <td>-8.831597e+08</td>
      <td>-8.852660e+08</td>
      <td>-9.669854e+08</td>
      <td>-9.095064e+08</td>
      <td>3.094863e+07</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.661567</td>
      <td>0.013049</td>
      <td>0.011522</td>
      <td>0.000856</td>
      <td>8</td>
      <td>10</td>
      <td>NaN</td>
      <td>{'max_features': 8, 'n_estimators': 10}</td>
      <td>-2.565200e+09</td>
      <td>-2.760625e+09</td>
      <td>...</td>
      <td>-2.680120e+09</td>
      <td>1.436614e+08</td>
      <td>5</td>
      <td>-5.009905e+08</td>
      <td>-4.870524e+08</td>
      <td>-4.837687e+08</td>
      <td>-5.273396e+08</td>
      <td>-5.073409e+08</td>
      <td>-5.012984e+08</td>
      <td>1.565241e+07</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2.099403</td>
      <td>0.083338</td>
      <td>0.035778</td>
      <td>0.004660</td>
      <td>8</td>
      <td>30</td>
      <td>NaN</td>
      <td>{'max_features': 8, 'n_estimators': 30}</td>
      <td>-2.336182e+09</td>
      <td>-2.611561e+09</td>
      <td>...</td>
      <td>-2.510836e+09</td>
      <td>1.374942e+08</td>
      <td>2</td>
      <td>-3.807474e+08</td>
      <td>-3.895396e+08</td>
      <td>-3.731072e+08</td>
      <td>-3.865464e+08</td>
      <td>-3.940694e+08</td>
      <td>-3.848020e+08</td>
      <td>7.274341e+06</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.110337</td>
      <td>0.005226</td>
      <td>0.005313</td>
      <td>0.000367</td>
      <td>2</td>
      <td>3</td>
      <td>False</td>
      <td>{'bootstrap': False, 'max_features': 2, 'n_est...</td>
      <td>-3.676333e+09</td>
      <td>-4.174187e+09</td>
      <td>...</td>
      <td>-4.050199e+09</td>
      <td>2.068671e+08</td>
      <td>17</td>
      <td>-0.000000e+00</td>
      <td>-0.000000e+00</td>
      <td>-0.000000e+00</td>
      <td>-0.000000e+00</td>
      <td>-0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.417751</td>
      <td>0.083123</td>
      <td>0.014098</td>
      <td>0.001016</td>
      <td>2</td>
      <td>10</td>
      <td>False</td>
      <td>{'bootstrap': False, 'max_features': 2, 'n_est...</td>
      <td>-2.679464e+09</td>
      <td>-2.985912e+09</td>
      <td>...</td>
      <td>-2.924522e+09</td>
      <td>1.304188e+08</td>
      <td>10</td>
      <td>-0.000000e+00</td>
      <td>-9.463245e+02</td>
      <td>-0.000000e+00</td>
      <td>-0.000000e+00</td>
      <td>-0.000000e+00</td>
      <td>-1.892649e+02</td>
      <td>3.785298e+02</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.148054</td>
      <td>0.008795</td>
      <td>0.005098</td>
      <td>0.000627</td>
      <td>3</td>
      <td>3</td>
      <td>False</td>
      <td>{'bootstrap': False, 'max_features': 3, 'n_est...</td>
      <td>-3.413029e+09</td>
      <td>-3.862638e+09</td>
      <td>...</td>
      <td>-3.678788e+09</td>
      <td>1.461245e+08</td>
      <td>16</td>
      <td>-0.000000e+00</td>
      <td>-0.000000e+00</td>
      <td>-0.000000e+00</td>
      <td>-0.000000e+00</td>
      <td>-0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.484264</td>
      <td>0.008298</td>
      <td>0.013995</td>
      <td>0.000664</td>
      <td>3</td>
      <td>10</td>
      <td>False</td>
      <td>{'bootstrap': False, 'max_features': 3, 'n_est...</td>
      <td>-2.743328e+09</td>
      <td>-2.828006e+09</td>
      <td>...</td>
      <td>-2.794703e+09</td>
      <td>5.917227e+07</td>
      <td>8</td>
      <td>-0.000000e+00</td>
      <td>-0.000000e+00</td>
      <td>-0.000000e+00</td>
      <td>-0.000000e+00</td>
      <td>-0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.178736</td>
      <td>0.007328</td>
      <td>0.005338</td>
      <td>0.000965</td>
      <td>4</td>
      <td>3</td>
      <td>False</td>
      <td>{'bootstrap': False, 'max_features': 4, 'n_est...</td>
      <td>-3.318034e+09</td>
      <td>-3.381403e+09</td>
      <td>...</td>
      <td>-3.352938e+09</td>
      <td>8.572118e+07</td>
      <td>12</td>
      <td>-1.962887e+02</td>
      <td>-0.000000e+00</td>
      <td>-0.000000e+00</td>
      <td>-1.051392e+04</td>
      <td>-0.000000e+00</td>
      <td>-2.142042e+03</td>
      <td>4.186630e+03</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.584409</td>
      <td>0.007127</td>
      <td>0.013667</td>
      <td>0.000872</td>
      <td>4</td>
      <td>10</td>
      <td>False</td>
      <td>{'bootstrap': False, 'max_features': 4, 'n_est...</td>
      <td>-2.593077e+09</td>
      <td>-2.810679e+09</td>
      <td>...</td>
      <td>-2.703044e+09</td>
      <td>1.018621e+08</td>
      <td>6</td>
      <td>-0.000000e+00</td>
      <td>-0.000000e+00</td>
      <td>-0.000000e+00</td>
      <td>-0.000000e+00</td>
      <td>-0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
  </tbody>
</table>
<p>18 rows × 23 columns</p>
</div>



## Other Methods to Fine Tune
* Randomized Search:
  * Iteration dependent so it can explore more options for hyperparameters
  * More control of computing budget
* Ensemble Methods:
  * Combining the models which perform the best

## Feature Importance


```python
attributes = list(df.columns) + list(encoder.classes_)
attributes.remove("median_house_value")
attributes.remove("ocean_proximity")
importances = grid_search.best_estimator_.feature_importances_
sorted(zip(importances, attributes), reverse=True)
```




    [(0.3317845401920315, 'median_income'),
     (0.14391320344674868, 'INLAND'),
     (0.10526089823364354, 'population_per_household'),
     (0.08263855622539133, 'bedrooms_per_room'),
     (0.08109436950269967, 'longitude'),
     (0.06119936528237925, 'latitude'),
     (0.05437513667126127, 'rooms_per_household'),
     (0.04269180191935387, 'housing_median_age'),
     (0.018543650605563098, 'population'),
     (0.017855965561009164, 'total_rooms'),
     (0.01747459825864214, 'total_bedrooms'),
     (0.016371631697584668, 'households'),
     (0.015137593949840484, '<1H OCEAN'),
     (0.006837130390816489, 'NEAR OCEAN'),
     (0.004801246718794319, 'NEAR BAY'),
     (2.0311344240502856e-05, 'ISLAND')]



## Evaluate Your Model on the Test Set


```python
final_model = grid_search.best_estimator_
print(final_model)

X_test = strat_test_set.drop(columns="median_house_value")
# Little bit of an oversight here
X_test["population_per_household"] = X_test["population"] / X_test["households"]
X_test["bedrooms_per_room"] = X_test["total_bedrooms"] / X_test["total_rooms"]
X_test["rooms_per_household"] = X_test["total_rooms"] / X_test["households"]
y_test = strat_test_set["median_house_value"].copy().values

X_test_prep = pipeline.transform(X_test)

final_preds = final_model.predict(X_test_prep)
final_mse = mean_squared_error(y_test, final_preds)
final_rmse = np.sqrt(final_mse)
print(final_rmse)
```

    RandomForestRegressor(max_features=6, n_estimators=30)
    48308.099325390474


## Launch, Monitor, and Maintain Your System
* Plug into production data
* Write tests
* Write monitoring code to check the live performance at regular intervals and trigger alerts when it drops
  * Models tend to "rot" over time
* Evaluating performance requires sampling the system's predictions and evaluating them
* In this case we need to have the human evaluation plugged into the system
* Monitor system input data
* Automate the training process and train on fresh data
* For online learning save snapshots at regular intervals
