---
layout: notes
chapter: 2
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

housing_url = "https://raw.githubusercontent.com/ageron/" +
              "handson-ml/master/datasets/housing/housing.tgz"

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
df.hist(bins=100, figsize=(16,12));
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
train_set, test_set = train_test_split(df, 
                                       test_size=0.2, 
                                       random_state=42)
```

Let's see how similar these are:


```python
ax = train_set.hist(bins=5, 
                    figsize=(16,12), 
                    density=True, 
                    alpha=0.8);
test_set.hist(bins=5, 
              figsize=(16,12), 
              ax=ax, 
              density=True, 
              alpha=0.8);
```


    
![png](/assets/images/ml-book/chapter2/notes_12_0.png)
    


That looks pretty good to me. But we'll also do the stratified method:


```python
# Sample to same count in each bin (5 bins)
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
strat_train_set, strat_test_set = train_test_split(df, 
                                                   test_size=0.2, 
                                                   random_state=42, 
                                                   stratify=strat)
ax = strat_train_set.hist(bins=5, 
                          figsize=(16,16), 
                          density=True, 
                          alpha=0.8);
strat_test_set.hist(bins=5, 
                    figsize=(16,16), 
                    ax=ax.flatten()[:-2], 
                    density=True, alpha=0.8);


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
strat_train_set, strat_test_set = train_test_split(df, 
                                                   test_size=0.2, 
                                                   random_state=42, 
                                                   stratify=strat)
ax = strat_train_set.hist(bins=5, figsize=(16,16), 
                          density=True, alpha=0.8);
strat_test_set.hist(bins=5, figsize=(16,16), 
                    ax=ax.flatten()[:-2], 
                    density=True, 
                    alpha=0.8);


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
sns.scatterplot(x="longitude", y="latitude", data=df, 
                s=df["population"] / 50, 
                hue=df["median_house_value"], 
                alpha=0.3, palette="seismic");

```


    
![png](/assets/images/ml-book/chapter2/notes_19_0.png)
    



```python

```
