# Scikit learn

Most algorithms in scikit-learn need data to be stored in a two-dimensional array or matrix. The arrays are usually numpy arrays.



Scikit-learn strives to have a uniform interface across all methods, and we'll see examples of these below. Given a scikit-learn estimator object named model, the following methods are available:

Available in all estimators

model.fit() : fit training data. For supervised learning applications, this accepts two arguments: the data X and the labels y (e.g. model.fit(X, y)). For unsupervised learning applications, this accepts only a single argument, the data X (e.g. model.fit(X)).

Available in supervised estimators

model.predict() : given a trained model, predict the label of a new set of data. This method accepts one argument, the new data X_new (e.g. model.predict(X_new)), and returns the learned label for each object in the array.

model.predict_proba() : For classification problems, some estimators also provide this method, which returns the probability that a new observation has each categorical label. In this case, the label with the highest probability is returned by model.predict().

model.score() : for classification or regression problems, most (all?) estimators implement a score method. Scores are between 0 and 1, with a larger score indicating a better fit.

Available in unsupervised estimators

model.predict() : predict labels in clustering algorithms.

model.transform() : given an unsupervised model, transform new data into the new basis. This also accepts one argument X_new, and returns the new representation of the data based on the unsupervised model.
model.fit_transform() : some estimators implement this method, which more efficiently performs a fit and a transform on the same input data.