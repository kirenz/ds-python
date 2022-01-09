# Data

## Data ingestion

Typically, you first have to ingest your data. This means that you take data stored in a file, a relational database, a NoSQL database or data lakehouse and load it into Python. 

:::{Note}
In our examples, we often simply use [pandas to import CSV files](https://kirenz.github.io/pandas/pandas-intro-short.html#read-and-write-data).
:::

There are various options how to manage your data and if you want to learn more about the basics of data engineering, like:

- basics of big data (Hadoop ecosystem and Spark),
- relational and NoSQL databases,
- how to set up a PostgreSQL and MySQL database,
- examples of different data architectures and
- components of machine learning operations (MLOPS),

 review this resource: 

 ```{admonition} Data engineering 
:class: tip

- [Introduction to Data Engineering](https://kirenz.github.io/data-engineering/docs/intro.html)

```

## Data splitting

Once you’ve imported your data, it is a good idea to split your data into a *training* and *test set* {cite:p}`Geron2019`: We do this because this is the only way to know how well a model will generalize to new cases. This means we train our model only on the training set, and we test it using the test set. 

:::{note}
We typically use scikit-learn's [train test split function](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) to perform data splitting
:::

The error rate on new cases is called the *generalization error* (or out-of-sample error), and by evaluating our model on the test set, we get an estimate of this error. This value tells you how well your model will perform on instances it has never seen before. If the training error is low (i.e., your model makes few mistakes on the training set) but the generalization error is high, it means that your model is **overfitting** the training data.

Note that if we want to evaluate different settings (“hyperparameters”) for models, such as the alpha in Lasso, there is still a risk of overfitting on the test set because the parameters can be tweaked until the model performs optimally ([skicit learn developers](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation)). This way, knowledge about the test set can “leak” into the model and evaluation metrics no longer report on generalization performance. To solve this problem, yet another part of the dataset can be held out as a so-called “**validation set**”: training proceeds on the *training set*, after which evaluation is done on the *validation set*, and when the experiment seems to be successful, final evaluation can be done on the *test set*.

However, by partitioning the available data into three sets, we drastically reduce the number of samples which can be used for learning the model, and the results can depend on a particular random choice for the pair of (train, validation) sets ([skicit learn developers](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation)). A solution to this problem is a procedure called cross-validation (CV for short). A test set should still be held out for final evaluation, but the validation set is no longer needed when doing CV. In the basic approach, called k-fold CV, the training set is split into k smaller sets .


<iframe src="https://docs.google.com/presentation/d/e/2PACX-1vTPAoobEeafrN7WzxPwwKBr4G18Yh3P12ru6b123FakIWspNXe6EJU47nBKjfBqs1S7U-2Jwdhm_RKD/embed?start=false&loop=false&delayms=3000" frameborder="0" width="840" height="530" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>

## Data analysis and preprocessing

The first task in any data science or ML project is to understand and clean the (training) data, which includes ([Google Developers, 2022](https://www.tensorflow.org/tfx/tutorials/tfx/penguin_tfdv)):

1. Analyze the data: Understanding the data types, distributions, and other information (e.g., mean value, or number of uniques) about each feature
1. Define schema: Generating a preliminary schema that describes the data (float; integer, ...)
1. Anomaly detection: Identifying anomalies and missing values in the data with respect to given schema

We analyze the training data to understand important predictor characteristics such as their individual distributions, the degree of missingness within each predictor, potentially unusual values within predictors, relationships between predictors, and the relationship between each predictor and the response and so on {cite:p}`Kuhn2019`. 

In particular, exploratory data analysis (EDA) is used to understand if there are any challenges associated with the data that can be discovered prior to modeling (like multicollinearity). Furthermore, good visualisations will show you things that you did not expect, or raise new questions about the data {cite:p}`Wickham2016`: A good visualisation might also hint that you’re asking the wrong question, or you need to collect different data. 


 ```{admonition} Exploratory data analysis 
:class: tip

- [Data analysis in pandas](https://kirenz.github.io/pandas/intro.html)
- [Data exploration with seaborn](https://seaborn.pydata.org/) 
- [From Data to Viz](https://www.data-to-viz.com/) leads you to the most appropriate graph for your data.

```

## Feature engineering

> "Applied machine learning is basically feature engineering" Andrew Ng 

The understanding gained from our data analysis is now used for feature engineering,  which is the process of using domain knowledge to extract meaningful features (attributes) from raw data. The goal of this process is to create new features which improve the predictions from our model and my include steps like
 
- Feature transformation (encoding of categorical features; transformation of continous features)
- Feature extraction (reduce the number of features by combining existing features)
- Feature creation (make new features)

Feature extraction can be achieved by simply using the ratio of two predictors or with the help of more complex methodologies like pricipal component analysis. Typically we als need to transform continuous predictors because predictors may {cite:p}`Kuhn2019`:

- be on vastly different scales.
- follow a skewed distribution where a small proportion of samples are orders of magnitude larger than the majority of the data (i.e., skewness).
- contain a small number of extreme values.
- be censored on the low and/or high end of the range.
- have a complex relationship with the response and is truly predictive but cannot be adequately represented with a simple function or extracted by sophisticated models.
- contain relevant and overly redundant information. That is, the information collected could be more effectively and efficiently represented with a smaller, consolidated number of new predictors while still preserving or enhancing the new predictors’ relationship with the response.

 ```{admonition} Feature engineering 
:class: tip

- [Feature extraction in scikit-learn](https://scikit-learn.org/stable/modules/feature_extraction.html)

- Review {cite:t}`Kuhn2019` for a detailed discusson of feature engineering methods.
```

## Pipelines in scikit-learn

scikit-learn provides a library of transformers for data preprocessing and feature engineering, which may 

- clean data (see [Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing), 
- reduce features (see [Unsupervised dimensionality reduction](https://scikit-learn.org/stable/modules/unsupervised_reduction.html#data-reduction)), or 
- extract features (see [Feature extraction](https://scikit-learn.org/stable/modules/feature_extraction.htmln)).

Just as it is important to test a model on data held-out from training, data preprocessing (such as standardization, etc.) and similar data transformations similarly should be learnt from a training set and applied to held-out data for prediction. Here, **Pipelines** are a best practice to help avoid leaking statistics from your test data into the trained model (e.g. during cross-validation).


 ```{admonition} Pipelines 
:class: tip

- [Regression example with preprocessing pipeline](https://kirenz.github.io/regression/docs/case-duke-sklearn.html)

- scikit-learn's [Pipelines documentation](https://scikit-learn.org/stable/modules/compose.html#pipeline)

```