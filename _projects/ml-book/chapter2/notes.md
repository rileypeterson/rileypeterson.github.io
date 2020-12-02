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

## Download the Data




