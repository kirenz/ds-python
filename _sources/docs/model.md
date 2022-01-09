# Model

Once our features have been encoded in a format ready for modeling algorithms, they can be used in the training of the model. Note that the process of analysis, feature engineering and modeling often requires multiple iterations. The general phases are {cite:p}`Kuhn2021`:

![](https://www.tmwr.org/premade/modeling-process.svg)

The colored segments within the circles signify the repeated data splitting (cross validation) used during model training. We discuss the model building process in the following sections.

## Select algorithm

One of the hardest parts during the data science lifecycle can be finding the right algorithm for the job since different algorithms are better suited for different types of data and different problems. For some datasets the best algorithm could be a linear model, while for other datasets it is a random forest or neural network. There is no model that is a priori guaranteed to work better. This fact is known as the *"No Free Lunch (NFL) theorem"* {cite:p}`Wolpert1996`.

<br>

<iframe src="https://docs.google.com/presentation/d/e/2PACX-1vTcSDvoljfuWHqUueJAghObDxNULvu-jWuiprqYeeMvA9tITk8gSis1qWsRSAGblEjkExoEiBXFvaPN/embed?start=false&loop=false&delayms=3000" frameborder="0" width="840" height="520" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>

<br>

Some of the most common algorithms are Linear and Polynomial Regression, Logistic Regression, k-Nearest Neighbors, Support Vector Machines, Decision Trees, Random Forests, Neural Networks and Ensemble methods like Gradient Boosted Decision Trees (GBDT). 

A model **ensemble**, where the predictions of multiple single learners are aggregated together to make one prediction, can produce a high-performance final model. The most popular methods for creating ensemble models in scikit-learn are: 

- [Bootstrap aggregating](https://scikit-learn.org/stable/modules/ensemble.html#bagging-meta-estimator), also called bagging (from bootstrap aggregating) 
- [random forest](https://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees) 
- Boosting: [AdaBoost](https://scikit-learn.org/stable/modules/ensemble.html#adaboost) and [Gradient Tree Boosting](https://scikit-learn.org/stable/modules/ensemble.html#gradient-tree-boosting)

Each of these methods combines the predictions from multiple versions of the same type of model (e.g., classifications trees).

Note that the only way to know for sure which model is best would be to evaluate them all {cite:p}`Geron2019`. Since this is often not possible, in practice you make some assumptions about the data and evaluate only a few reasonable models. For example, for simple tasks you may evaluate linear models with various levels of regularization as well as some ensemble methods like Gradient Boosted Decision Trees (GBDT). For very complex problems, you may evaluate various deep neural networks.

The following flowchart was provided by scikit-learn developers to give users a bit of a rough guide on how to approach problems with regard to which algorithms to try on your data:

<br>

```{image} ../_static/img/algorithms.png
:alt: datascience
:class: bg-primary mb-1
:width: 600px
:align: center
```

<br>

Visit [this site](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html) to interact with the flowchart.

## Train and evaluate

In the first phase of the model building process, a variety of initial models are generated and their performance is compared during model evaluation. In model evaluation, we mainly assess the model’s performance metrics and examine residual plots to understand how well the models work. Scikit-learn provides an extensive list of possible metrics to quantify the quality of model predictions:

```{admonition} Metrics 
:class: tip

- [Metrics and scoring: quantifying the quality of predictions](https://scikit-learn.org/stable/modules/model_evaluation.html)

```

Our first goal in this process is to shortlist a few (two to five) promising models. 

## Tuning

After we identified a shortlist of promising models, it usually makes sense to tune the hyper-paramters of our models. 

:::{Note}
Hyper-parameters are parameters that are not directly learnt within algorithms. 
:::

In scikit-learn, hyper-paramters are passed as arguments to the algorithm like "alpha" for Lasso or "K" for the number of neighbors in a K-nearest neighbors model. 

Instead of trying to find good hyper-paramters manually, it is recommended to search the hyper-parameter space for the best cross validation score using one of the two generic approaches provided in scikit-learn:

- for given values, `GridSearchCV` exhaustively considers all parameter combinations.
- `RandomizedSearchCV` can sample a given number of candidates from a parameter space with a specified distribution.

```{admonition} Tuning 
:class: tip

- [Tuning the hyper-parameters of an estimator](https://scikit-learn.org/stable/modules/grid_search.html)
```

The *GridSearchCV* approach is fine when you are exploring relatively few combinations, but when the hyperparameter search space is large, it is often preferable to use *RandomizedSearchCV* instead {cite:p}`Geron2019`. Both methods use cross-validation (CV) to evaluate combinations of hyperparameter values. 

## Voting and stacking

Note that it often makes sense to combine different models since the group will usually perform better than the best individual model, especially if the individual models make very different types of errors.

Model **voting** simply combines the predictions for multiple models of any type. scikit-learn provides voting methods for both classification ([VotingClassifier](https://scikit-learn.org/stable/modules/ensemble.html#voting-classifier)) and regression ([VotingRegressor](https://scikit-learn.org/stable/modules/ensemble.html#voting-regressor)). 

:::{note}
Voting can be useful for a set of equally well performing models in order to balance out their individual weaknesses.
:::

In *classification* problems, the idea behind voting is to combine conceptually different machine learning classifiers and use a majority vote or the average predicted probabilities (soft vote) to predict the class labels. In *regression* problems, we combine different machine learning regressors and return the average predicted values. 

**Stacked** generalization is a method for combining estimators to reduce their biases. Therefore, the predictions of each individual estimator are stacked together and used as input to a final estimator to compute the prediction. This final estimator is trained through cross-validation.

In scikit-learn, the [StackingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html#sklearn.ensemble.StackingClassifier) and [StackingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html#sklearn.ensemble.StackingRegressor) provide such strategies which can be applied to classification and regression problems.

:::{note}
Model stacking is an ensembling method that takes the outputs of many models and combines them to generate a new model that generates predictions informed by each of its members.
:::

To learn more about the concept of stacking, visit the [documentation of stacks](https://stacks.tidymodels.org), a R package for model stacking. 

## Evaluate best model

After we tuned hyper-parameters and/or performed voting/stacking, we evaluate the best model (system) and their errors in detail. 

In particular, we take a look at the specific errors that our model (system) makes, and try to understand why it makes them and what could fix the problem - like adding extra features or getting rid of uninformative ones, cleaning up outliers, etc. {cite:p}`Geron2019`. If possible, we also display the importance scores of our predictors (e.g. using scikit-learn's [permutation feature importance](https://scikit-learn.org/stable/modules/permutation_importance.html)). With this information, we may want to try dropping some of the less useful features to make sure our model generalizes well.

After evaluating the model (system) for a while, we eventually have a system that performs sufficiently well.

## Evaluate on test set

Now is the time to evaluate the final model on the test set. If you did a lot of hyperparameter tuning, the performance will usually be slightly worse than what you measured using cross-validation - because your system ends up fine-tuned to perform well on the validation data and will likely not perform as well on unknown dataset {cite:p}`Geron2019`.

It is important to note that we don't change the model (system) anymore to make the numbers look good on the test set; the improvements would be unlikely to generalize to new data. Instead, we use the metrics for our final evaluation to make sure the model performs sufficiently well regarding our success metrics from the planning phase.