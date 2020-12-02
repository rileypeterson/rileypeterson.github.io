---
layout: exercises
chapter: 2
permalink: /ml-book/chapter2/exercises.html
---
1. Machine learning is the process of enabling machines (i.e. computers) to solve problems without being explicitly programs. This is done by providing data and letting the machine "decide" the optimal solution to the problem (via some loss function).
2. It shines when solutions exist to a problem but they are too complicated to explicitly program, when solutions don't actually exist to a problem, when you expect incoming data to change and you want a solution which is robust enough to adapt, and when you want to gain insight into the solution (i.e. dissect the blackbox).
3. When you give the model the result (i.e. label) that you expect it to predict.
4. Classification and a numerical prediction.
5. Anomaly detection, dimensionality reduction, association rule learning, clustering.
6. Reinforcement learning
7. Clustering algorithm
8. Spam is a supervised learning problem.
9. When a model's parameters actively change on new data.
10. When you use online learning because data can't fit into memory.
11. Instance-based model
12. The learning algorithm's hyperparameter stays constant throughout the training whereas the model parameter changes.
13. They search for an equation or non-linear combination of outputs which gives an accurate prediction. They tune parameters by minimizing a utility function. To make predictions you input the new data into the model and it gives you a prediction.
14. Overfitting, lousy data, underfitting, and not having a sufficient amount of data.
15. Overfitting. Moar data, impose regularization, provide a more representative sample of training data.
16. A test set gives some indication of the accuracy of the _production_ model (i.e. has never "seen" the incoming samples).
17. The validation set allows you to tweak hyperparameters of the model while still maintaining the purity of the test set.
18. The model may not generalize well when put in production since hyperparameters were fitted/chosen according to the test set. You'd probably see worse performance in production.
19. I don't think cross validation was very well explained. You make a test/train split and instead of splitting the training data again into a validation/train split, you pass over all the data multiple times, each time using a different (non-overlapping) portion for validation. You can average the validation loss over all the iterations. In this way your model is trained on all the data in the train set, while still reaping the benefits of having a validation set (hyperparameter tuning). 

<img src="https://upload.wikimedia.org/wikipedia/commons/b/b5/K-fold_cross_validation_EN.svg" style="background: white; margin-left: auto; margin-right: auto; display: block;">
