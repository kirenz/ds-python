# Model

Once our features have been encoded in a format ready for modeling algorithms, they can be used in the training of the model. Note that the process of analysis, feature engineering and modeling often requires multiple iterations. The general phases are {cite:p}`Kuhn2021`:

![](https://www.tmwr.org/premade/modeling-process.svg)

The colored segments within the circles signify the repeated data splitting and usage of crossvalidation used during resampling. We discuss the model building process in the following sections.

## Select algorithm

One of the hardest parts during the data science lifecycle can be finding the right algorithm for the job since different algorithms are better suited for different types of data and different problems. For some datasets the best algorithm could be a linear model, while for other datasets it is a random forest or neural network. There is no model that is a priori guaranteed to work better. This fact is known as the "No Free Lunch (NFL) theorem" {cite:p}`Wolperet1996`.

Some of the most common algorithms are Linear and Polynomial Regression, Logistic Regression, k-Nearest Neighbors, Support Vector Machines, Decision Trees, Random Forests, Neural Networks and Ensemble methods like Gradient Boosted Decision Trees (GBDT). A model **ensemble**, where the predictions of multiple single learners are aggregated together to make one prediction, can produce a high-performance final model. The most popular methods for creating ensemble models are bagging (Breiman, 1996), random forest (Ho 1995; Breiman 2001), and boosting (Freund and Schapire 1997). Each of these methods combines the predictions from multiple versions of the same type of model (e.g., classifications trees).

The only way to know for sure which model is best would be to evaluate them all {cite:p}`Geron2019`. Since this is often not possible, in practice you make some assumptions about the data and evaluate only a few reasonable models. For example, for simple tasks you may evaluate linear models with various levels of regularization as well as some ensemble methods like Gradient Boosted Decision Trees (GBDT). For very complex problems, you may evaluate various deep neural networks.

The following interactive flowchart was provided by scikit-learn developers to give users a bit of a rough guide on how to approach problems with regard to which algorithms to try on your data ([scikit-learn](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)):


<div>
  <iframe id="scikit-learn"
      title="Choosing the right estimator"
      width="960"
      height="569"
      src="https://scikit-learn.org/stable/_static/ml_map.png">
  </iframe

## Train and evaluate

In the first phase of the model building process, a variety of initial models are generated and their performance is compared during model evaluation. In model evaluation, we assess the model’s performance metrics, examine residual plots, and conduct other EDA-like analyses to understand how well the models work {cite:p}`Kuhn2019`. Scikit-learn provides an extensive list of possible metrics to quantify the quality of model predictions:

- [Metrics and scoring: quantifying the quality of predictions](https://scikit-learn.org/stable/modules/model_evaluation.html)

Our first goal in this process is to shortlist a few (two to five) promising models {cite:p}`Geron2019`. 

## Tuning

After we identified a shortlist of promising models, it usually makes sense to tune the hyper-paramters of our models. Hyper-parameters are parameters that are not directly learnt within the modeling process. In scikit-learn they are passed as arguments to the algorithm (like alpha for Lasso or K for the number of neighbors) in a K-nearest neighbors model). 

Instead of trying to find good hyper-paramters manually, it is recommended to search the hyper-parameter space for the best cross validation score using methods from scikit-learn. Two generic approaches to parameter search are provided in scikit-learn: 

- [Tuning the hyper-parameters of an estimator](https://scikit-learn.org/stable/modules/grid_search.html)

  - for given values, `GridSearchCV` exhaustively considers all parameter combinations
  - `RandomizedSearchCV` can sample a given number of candidates from a parameter space with a specified distribution

The grid search approach is fine when you are exploring relatively few combinations, but when the hyperparameter search space is large, it is often preferable to use RandomizedSearchCV instead {cite:p}`Geron2019`. Both methods use cross-validation to evaluate combinations of hyperparameter values. 

 and/or try to combine the models that perform best. Note that it often makes sense to combine different models since the group (or “ensemble”) will often perform better than the best individual model, especially if the individual models make very different types of errors. The goal of ensemble methods is to combine the predictions of several base estimators built with a given learning algorithm in order to improve generalizability / robustness over a single estimator.

## Stacking

Model stacking combines the predictions for multiple models of any type. For example, a logistic regression, classification tree, and support vector machine can be included in a stacking ensemble.

Scikit-learn provides stacking methods for both classification (VotingClassifier) and regression (VotingRegressor):

In classification settings, The idea behind the VotingClassifier is to combine conceptually different machine learning classifiers and use a majority vote or the average predicted probabilities (soft vote) to predict the class labels. Such a classifier can be useful for a set of equally well performing model in order to balance out their individual weaknesses.



### Analyze best models

After we tuned the hyperparameter, we analyze the best models and their errors.

Let’s display these importance scores next to their corresponding attribute names:

You will often gain good insights on the problem by inspecting the best models. For example, the RandomForestRegressor can indicate the relative importance of each attribute for making accurate predictions:

### Evaluate final model on test set

After tweaking your models for a while, you eventually have a system that performs sufficiently well. Now is the time to evaluate the final model on the test set. There is nothing special about this process; just get the predictors and the labels from your test set, run your full_pipeline to transform the data (call transform(), not fit_transform()—you do not want to fit the test set!), and evaluate the final model on the test set:


If you did a lot of hyperparameter tuning, the performance will usually be slightly worse than what you measured using cross-validation (because your system ends up fine-tuned to perform well on the validation data and will likely not perform as well on unknown datasets). It is not the case in this example, but when this happens you must resist the temptation to tweak the hyperparameters to make the numbers look good on the test set; the improvements would be unlikely to generalize to new data.

Now comes the project prelaunch phase: you need to present your solution (highlighting what you have learned, what worked and what did not, what assumptions were made, and what your system’s limitations are), document everything, and create nice presentations with clear visualizations and easy-to-remember statements (e.g., “the median income is the number one predictor of housing prices”). In this California housing example, the final performance of the system is not better than the experts’ price estimates, which were often off by about 20%, but it may still be a good idea to launch it, especially if this frees up some time for the experts so they can work on more interesting and productive tasks.


XX OLD





- Exploratory data analysis (EDA): Initially there is a back and forth between numerical analysis and visualization of the data  where different discoveries lead to more questions and data analysis “side-quests” to gain more understanding.

- Feature engineering: The understanding gained from EDA results in the creation of specific model terms that make it easier to accurately model the observed data. This can include complex methodologies (e.g., PCA) or simpler features (using the ratio of two predictors). 

- Model tuning and selection (circles with blue and yellow segments): A variety of models are generated and their performance is compared. Some models require parameter tuning where some structural parameters are required to be specified or optimized. The colored segments within the circles signify the repeated data splitting used during resampling.

- Model evaluation: During this phase of model development, we assess the model’s performance metrics, examine residual plots, and conduct other EDA-like analyses to understand how well the models work. 

The last step is **communication**, an absolutely critical part of any data analysis project. It doesn't matter how well your models and visualisation have led you to understand the data unless you can also communicate your results to others.

Surrounding all the data science steps covered above is **programming**. Programming is a cross-cutting tool that you use in every part of the project. You don’t need to be an expert programmer to be a data scientist, but learning more about programming pays off because becoming a better programmer allows you to automate
common tasks, and solve new problems with greater ease.


{cite:p}`Wickham2016`
