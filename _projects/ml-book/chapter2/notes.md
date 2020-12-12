---
layout: notes
chapter: 2
permalink: /ml-book/chapter2/notes.html
---

## End-to-End Machine Learning Project
Machine learning is the process of enabling a machine to solve a particular problem without being explicitly programmed.

## The Process
* Big Picture (Problem Statement)
* Get Data
* Exploratory Data Analysis
* Data Preparation
* Model selection and training
* Fine-tune the model
* Production
* Monitor and Maintain

## The Task
Build a model to predict housing prices in California given California census data. Specifically, predict the median housing price in any district, given all other metrics.  

This is a supervised learning problem. It's a regression task. We can probably get away with using batch learning, since 
census data comes out once every 10(?) years and housing prices change pretty slowly over time.


## RMSE (Root Mean Squared Error)
Measures the standard deviation of the errors the system makes in its predictions. Recall the standard deviation is:

$$ \sigma = \sqrt{\frac{\sum_{i}{(\bar{X} - X_i)^2}}{N}} $$

Analogously RMSE is:

$$ RMSE = \sqrt{\frac{\sum_{i}{(y_i - f(X_i))^2}}{N}} $$

where $$ f $$ is our model. There is also Mean Absolute Error (MAE). RMSE is more 
sensitive to outliers than MAE because for large outliers (i.e. differences) 
RMSE will make them larger by squaring them.

## Environment Setup
I'm not really sure I want to use a jupyter notebook. More of a qtconsole kind of guy if I'm doing something interactive.
I think I'll make a little "prerequisite" snippet at the top of every file. For me, I really like having a full snippet, e.g. you
don't run into a NameError because of something defined in cell 2. Just want to be able to copy/paste and have it run :) Rant over

On second thought it might just be easier... we'll see. [This](https://github.com/jhrmnn/knitj) looks cool... might change that up a bit to kind of
make a custom frontend jupyter notebook which naturally interfaces with this website

## Download the Data
Geron puts the data in tarfiles. I wrote an ephemeral tarfile downloader to dataframe:

*code here*

## Checking Out the Data
Best way to do this is `df.hist` as Geron suggests. 

## Train Test Split Takeaways
Geron points out that in some cases it's beneficial to make sure you're getting the same samples
in the test set each time. Seems to me that a good model performs about the same on any test set, but if you want to do this
you can hash the values from each row and select 20% (or whatever) of them to form the test set. Since the dataset it static 
this doesn't seem too pertinent, but it's good to keep in mind. For now I'm just using sklearn `train_test_split` with random state.


## Stratified Sampling
You want to select your test set to be proportionally similar to the total population. Example: A particular district in Alaska is 75% Democratic and 25% Republican, if you're sample size is small you might not get that same proportion in your test set, which would be problematic. Again, I think another way to address this is creating a model which generalizes well to "any" random train/test split...

Geron goes on to perform a `StratifiedShuffleSplit` of the data. Figure 2-10 is an interesting illustration of how this can have a sizable effect. Rather than receiving a tip from "[chatting] with experts", let's suppose we didn't know which feature would be most important. We could probably deduce that same conclusion by looking at correlations. He discusses that in a bit.

## Visualization
I don't know if looking at the long/lat vs. median housing value really revealed too much that wasn't already known. 

## Correlations
This is important! Definitely need to always do this when exploring a dataset. It's important to note that correlations are only measuring 
the *linear* correlations, Figure 2-14 elucidates this.

## Imputer
The best practice way to fill NaN with median or other strategy. Page 60.

## Handling Categorical Attributes
Use `sklearn.preprocessing.LabelEncoder` or `sklearn.preprocessing.OneHotEncoder` (should be 2-D). The combo of these is `sklearn.preprocessing.LabelBinarizer`.

