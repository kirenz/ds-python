# Model

Once our features have been preprocessed in a format ready for modeling algorithms (see [](data.md)), they can be used in the training and selection of the model. Note that the type of preprocessing is dependent on the type of model being fit. {cite:t}`Kuhn2021` provide recommendations for baseline levels of preprocessing that are needed for various model functions (see [this table](https://www.tmwr.org/pre-proc-table.html)).

The process of analysis, data preprocessing, feature engineering, feature selection, modeling and model selection often requires multiple iterations. The general phases are {cite:p}`Kuhn2021`:

![](https://www.tmwr.org/premade/modeling-process.svg)

The colored segments within the circles signify the repeated data splitting (cross validation) used during model training. 

The following resources provide more detailed information about different regression and classifiaction models. 

```{admonition} Jupyter Book
:class: tip

- [Introduction to Regression](https://kirenz.github.io/regression/docs/intro.html)
- [Introduction to Classification](https://kirenz.github.io/classification/docs/intro.html)

```

Next, we discuss some important model selection topics like

- Model selection
- best fitting model 
- mean squared error
- bias-variance trade off.

<br>

<iframe src="https://docs.google.com/presentation/d/e/2PACX-1vRWlyTZB6YpdYyRpWXdaI5_s8o9MZ5DFk9Gm-cTO4CrrJBHrNgrcyZl4IdktJEMq0e4apMDPpMP46Cb/embed?start=false&loop=false&delayms=3000" frameborder="0" width="820" height="520" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>

<br>

```{admonition} Resources
:class: tip
- [Download slides](https://docs.google.com/presentation/d/1ZrzKUPZqp7GlCAh4uFpXnvUW6kQTvUgD2G-YZ7B24aM/export/pdf)
- Colab: [Regression example: Does money make people happier?](https://colab.research.google.com/github/kirenz/data-science-projects/blob/master/ds-first-steps-happy-gdp.ipynb)

```

In the next sections, we'll discuss the process of model building in detail.  

## Select algorithm

One of the hardest parts during the data science lifecycle can be finding the right algorithm for the job since different algorithms are better suited for different types of data and different problems. For some datasets the best algorithm could be a linear model, while for other datasets it is a random forest or neural network. There is no model that is a priori guaranteed to work better. This fact is known as the *"No Free Lunch (NFL) theorem"* {cite:p}`Wolpert1996`.

<br>

<iframe src="https://docs.google.com/presentation/d/e/2PACX-1vTcSDvoljfuWHqUueJAghObDxNULvu-jWuiprqYeeMvA9tITk8gSis1qWsRSAGblEjkExoEiBXFvaPN/embed?start=false&loop=false&delayms=3000" frameborder="0" width="820" height="520" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>


<br>

```{admonition} Resources
:class: tip
- [Download slides](https://docs.google.com/presentation/d/1ZycVKQLPSHGUv3Mga1CE3BHTL0W_gfszs6SkEh7fzeU/export/pdf)

```

Some of the most common algorithms are (take a look at the Jupyter Books [Regression](https://kirenz.github.io/regression/docs/intro.html) and [Classification](https://kirenz.github.io/classification/docs/intro.html) for more details):

- Linear and Polynomial Regression, 
- Logistic Regression, 
- k-Nearest Neighbors, 
- Support Vector Machines,
- Decision Trees, 
- Random Forests, 
- Neural Networks and
- Ensemble methods like Gradient Boosted Decision Trees (GBDT). 

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
:width: 800px
:align: center
```

<br>

Visit [this site](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html) to interact with the flowchart.

## Train and evaluate

In the first phase of the model building process, a variety of initial models are generated and their performance is compared during model evaluation. As a part of this process, we also need to decide which features we want to include in our model ("feature selection"). Therefore, let's first take a look at the topic of feature selection.

### Feature selection

There are a number of different strategies for feature selection that can be applied and some of them are performed simultaneously with model building. 

:::{Note}
Feature selection is the process of selecting a subset of relevant features (variables, predictors) for our model.
:::

If you want to learn more about feature selection methods, review the following content:

```{admonition} Jupyter Book 
:class: tip

- [Feature Selection](https://kirenz.github.io/feature-engineering/docs/feature-selection.html#)

```

### Training

Now we can use the pipeline we created in [](data.md) (see last section) and combine it with scikit-learn algorithms of our choice:


```Python
from sklearn.linear_model import LinearRegression

# Use pipeline with linear regression model
lm_pipe = Pipeline(steps=[
            ('full_pipeline', full_pipeline),
            ('lm', LinearRegression())
                         ])
```

```Python
# Show pipeline as diagram
set_config(display="diagram")

# Fit model
lm_pipe.fit(X_train, y_train)
```

### Evaluation

In model evaluation, we mainly assess the model’s performance metrics (using an evaluation set) and examine residual plots (see this [example for linear regression dagnostics](https://kirenz.github.io/regression/docs/diagnostics.html)) to understand how well the models work. 

Scikit-learn provides an extensive list of possible metrics to quantify the quality of model predictions:

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

Example of hyperparameter tuning with skicit-learn pipeline ([scikit-learn developers](https://scikit-learn.org/stable/tutorial/statistical_inference/putting_together.html)): Define a pipeline to search for the best combination of PCA truncation and classifier regularization.

```Python
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# Pipeline
pipe = Pipeline(steps=[
    ("scaler", StandardScaler()), 
    ("pca", PCA()), 
    ("logistic", LogisticRegression(max_iter=10000, tol=0.1))])

# Data
X_digits, y_digits = datasets.load_digits(return_X_y=True)

# Parameters of pipelines can be set using ‘__’ separated parameter names:
param_grid = {
    "pca__n_components": [5, 15, 30, 45, 60],
    "logistic__C": np.logspace(-4, 4, 4),
}

# Gridsearch
search = GridSearchCV(pipe, param_grid, n_jobs=2)
search.fit(X_digits, y_digits)

# Show results
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)
```

## Voting and stacking

It often makes sense to combine different models since the group ("ensemble") will usually perform better than the best individual model, especially if the individual models make very different types of errors. 

:::{note}
Voting can be useful for a set of equally well performing models in order to balance out their individual weaknesses.
:::

Model **voting** combines the predictions for multiple models of any type and thereby creating an ensemble meta-estimator. scikit-learn provides voting methods for both classification ([VotingClassifier](https://scikit-learn.org/stable/modules/ensemble.html#voting-classifier)) and regression ([VotingRegressor](https://scikit-learn.org/stable/modules/ensemble.html#voting-regressor)): 


- In *classification* problems, the idea behind voting is to combine conceptually different machine learning classifiers and use a majority vote or the average predicted probabilities (soft vote) to predict the class labels. 
- In *regression* problems, we combine different machine learning regressors and return the average predicted values. 


```{admonition} Voting regressor
:class: tip

- [Voting regressor example](https://kirenz.github.io/regression/docs/ensemble.html)
```

**Stacked** generalization is a method for combining estimators to reduce their biases. Therefore, the predictions of each individual estimator are stacked together and used as input to a final estimator to compute the prediction. This final estimator is trained through cross-validation.

:::{note}
Model stacking is an ensembling method that takes the outputs of many models and combines them to generate a new model that generates predictions informed by each of its members.
:::

In scikit-learn, the [StackingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html#sklearn.ensemble.StackingClassifier) and [StackingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html#sklearn.ensemble.StackingRegressor) provide such strategies which can be applied to classification and regression problems.


To learn more about the concept of stacking, visit the [documentation of stacks](https://stacks.tidymodels.org), a R package for model stacking. 

## Evaluate best model

After we tuned hyper-parameters and/or performed voting/stacking, we evaluate the best model (system) and their errors in detail. 

In particular, we take a look at the specific errors that our model (system) makes, and try to understand why it makes them and what could fix the problem - like adding extra features or getting rid of uninformative ones, cleaning up outliers, etc. {cite:p}`Geron2019`. If possible, we also display the importance scores of our predictors (e.g. using scikit-learn's [permutation feature importance](https://scikit-learn.org/stable/modules/permutation_importance.html)). With this information, we may want to try dropping some of the less useful features to make sure our model generalizes well.

After evaluating the model (system) for a while, we eventually have a system that performs sufficiently well.

## Evaluate on test set

Now is the time to evaluate the final model on the test set. If you did a lot of hyperparameter tuning, the performance will usually be slightly worse than what you measured using cross-validation - because your system ends up fine-tuned to perform well on the validation data and will likely not perform as well on unknown dataset {cite:p}`Geron2019`.

It is important to note that we don't change the model (system) anymore to make the numbers look good on the test set; the improvements would be unlikely to generalize to new data. Instead, we use the metrics for our final evaluation to make sure the model performs sufficiently well regarding our success metrics from the planning phase.

## Challenges

In the following presentation, we cover some typical modeling challenges:

- Poor quality data
- irrelevant features and feature engineering
- overfitting and regularization
- underfitting

<br>

<iframe src="https://docs.google.com/presentation/d/e/2PACX-1vQz4smNkQ4Ef0JL2RvMXqlb4RiagKxajxF_QekQhdq8czpX456ly7GgoLKk-tZ5khHSP6J6ztTjMs6X/embed?start=false&loop=false&delayms=3000" frameborder="0" width="820" height="520" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>

<br>

```{admonition} Slides
:class: tip
- [Download slides](https://docs.google.com/presentation/d/1WPUVfUe4rZu1bt61IwjneHqldhH9nC6U0F_u7Mj_ZL4/export/pdf)
```