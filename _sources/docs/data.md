# Data

## Data ingestion

Typically, you first have to ingest your data. This means that you take data stored in a file, a relational database, a NoSQL database or data lakehouse and load it into Python. Note that there are various options how to manage your data and if you want to learn more about the basics of data engineering, like:

- basics of big data (Hadoop ecosystem and Spark),
- relational and NoSQL databases,
- how to set up a PostgreSQL and MySQL database,
- examples of different data architectures and
- components of machine learning operations (MLOPS),

 review this resource: 

- [Introduction to Data Engineering](https://kirenz.github.io/data-engineering/docs/intro.html)

:::{note}
In our examples, we often use [pandas to import CSV files](https://kirenz.github.io/pandas/pandas-intro-short.html#read-and-write-data)
:::

## Data splitting

Once you’ve imported your data, it is a good idea to split your data into a *training* and *test set* {cite:p}`Geron2019`: We do this because this is the only way to know how well a model will generalize to new cases - therefore, we actually try it out on new cases (on our test data). This means we train our model using the training set, and we test it using the test set. 

The error rate on new cases is called the generalization error (or out-of-sample error), and by evaluating our model on the test set, we get an estimate of this error. This value tells you how well your model will perform on instances it has never seen before. If the training error is low (i.e., your model makes few mistakes on the training set) but the generalization error is high, it means that your model is overfitting the training data.

## Data analysis and preprocessing

The first task in any data science or ML project is to understand and clean the (training) data, which includes ([Google Developers, 2022](https://www.tensorflow.org/tfx/tutorials/tfx/penguin_tfdv)):

1. Analyze the data: Understanding the data types, distributions, and other information (e.g., mean value, or number of uniques) about each feature
1. Define schema: Generating a preliminary schema that describes the data (float; integer, ...)
1. Anomaly detection: Identifying anomalies and missing values in the data with respect to given schema

We analyze the training data to understand important predictor characteristics such as their individual distributions, the degree of missingness within each predictor, potentially unusual values within predictors, relationships between predictors, and the relationship between each predictor and the response and so on {cite:p}`Kuhn2019`. In particular, exploratory data analysis (EDA) is used to understand if there are any challenges associated with the data that can be discovered prior to modeling (like multicollinearity). Furthermore, good visualisations will show you things that you did not expect, or raise new questions about the data {cite:p}`Wickham2016`: A good visualisation might also hint that you’re asking the wrong question, or you need to collect different data. 


## Feature engineering

The understanding gained from our data analysis is now used (in combination with domain knowledge) for feature engineering. The goal of this process is to make it easier to accurately model the data. This process may include encoding of categorical predictors and the creation of new features by simply using the ratio of two predictors or with the help of more complex methodologies like pricipal component analysis.

Typically we als need to handle continuous predictors because predictors may {cite:p}`Kuhn2019`:

- be on vastly different scales.
- follow a skewed distribution where a small proportion of samples are orders of magnitude larger than the majority of the data (i.e., skewness).
- contain a small number of extreme values.
- be censored on the low and/or high end of the range.
- have a complex relationship with the response and is truly predictive but cannot be adequately represented with a simple function or extracted by sophisticated models.
- contain relevant and overly redundant information. That is, the information collected could be more effectively and efficiently represented with a smaller, consolidated number of new predictors while still preserving or enhancing the new predictors’ relationship with the response.
