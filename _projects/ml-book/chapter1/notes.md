---
layout: notes
chapter: 1
permalink: /ml-book/chapter1/notes.html
---

_Most of this we already know..._

## What is machine learning?

* Machine learning is the process of enabling a machine to solve a particular problem without being explicitly programmed.

## Why use machine learning?

* Existing solutions to a problem are vast and tedious to program
* There is no known solution or the solution is sufficiently complicated to be solved programmatically/ analytically
* You expect data to change. Machine learning models are capable of adapting to new data.

## What are the different types of machine learning?

* <ins>Supervised Learning</ins> - features of data go in along with labelled output, training occurs, and the model predicts the output give new input data
* Unsupervised learning - samples are not labelled, think clustering algorithms etc.
* Semisupervised learning - A few of the samples are labelled (e.g. unlabelled samples could be classified as belonging to the same cluster as the labelled sample)
* Reinforcement learning - An agent observes an environment and learns to perform actions which optimize for rewards over penalties

## Methods for learning

* Batch learning - An entire dataset is trained on, the model is launched into production, and it simply makes predictions on the new incoming data. The model doesn't change/learn anymore, unless it is pulled offline and trained again on a new set of data. Becomes difficult if you have a lot of data or data is varying often.
* Online learning - Learning occurs sequentially by consuming mini-batches of data. Drawbacks of this include bad data (where the model adjusts to quickly and performs poorly) or a learning rate which is too slow (model doesn't adjust quick enough to new data)

## Different ways the machine learning models generalize

* Instance-based learning - Memorize learning instances (i.e. samples), determine how similar a new instance is to existing data, act accordingly
* Model-based learning - You decide an equation, learn the free parameters of that equation, and then use the equation to make predictions on new data. Think linear regression

## Challenges of machine learning

* Insufficient data - If you don't have enough data your model won't generalize well
* Nonrepresentative training data - Obviously if you don't give it data similar to what you want it to predict, it won't stand a chance.
* Sampling bias - The data you collect is biased. That would be like me feeding in only data from only west coast teams to a sports algorithm because those are teams I care about the most (_for instance_). 
* Lousy data - Errors/inconsistencies in training data.
* Irrelevant features - You feed in data that does not impact the result. Feature engineering is the practice of discerning which features/data to use.
* Overfitting - Model doesn't generalize to new data because it "memorizes" a perfect/near-perfect model for training data
* Underfitting

## Train/Test Split

* Leave behind some of the training data to test as input into your model. This will give you insight into how your model generalizes. A common split is 80/20, train/test. If you are trying to determine the best hyperparameters it is useful to also make a 3rd partition of the data known as the validation set. That way you are not selecting hyperparameters based on the test set. 
