# Data Science Lifecycle

 From a high-level perspective, a typical data science project looks something like this :

```{image} ../_static/img/lifecycle-data.png
:alt: datascience
:class: bg-primary mb-1
:width: 600px
:align: center
```

<br>

Note that in this book, we will only cover the first three stages of the data science lifecycle.

## Plan

First, you have to define a plan of what you want to achive with your data science project. To do this, we start with the business model, which describes the rationale of how your organization creates, delivers and captures value. The complete process includes the following topics: 

:::{note}

1. Identify use case: Use the business model canvas.
2. Frame the problem: Provide a statement of what is to be learned and how decisions should be made.
3. Identify variables or labels: for structured data problems, we need to identify potentially relevant variables; for unstructured data problems, we need to define labels.
4. Define success metrics: Write down your metrics for success and failure with the data science project. 
:::

To learn more about the data science planning phase, review this presentation: 

<iframe src="https://docs.google.com/presentation/d/e/2PACX-1vR3mAfcepfacMwk7_ob-uPjSX6aMLISTxC2C1DEOyMS5HdO1RSY8fSbBdPP21JjKP0fHKoE46719xjJ/embed?start=false&loop=false&delayms=3000" frameborder="0" width="960" height="569" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>

## Data

### Data ingestion

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

### Data splitting

Once you’ve imported your data, it is a good idea to split your data into a *training* and *test set* {cite:p}`Geron2019`: We do this because this is the only way to know how well a model will generalize to new cases - therefore, we actually try it out on new cases (on our test data). This means we train our model using the training set, and we test it using the test set. 

The error rate on new cases is called the generalization error (or out-of-sample error), and by evaluating our model on the test set, we get an estimate of this error. This value tells you how well your model will perform on instances it has never seen before. If the training error is low (i.e., your model makes few mistakes on the training set) but the generalization error is high, it means that your model is overfitting the training data.

### Data analysis and preprocessing

The first task in any data science or ML project is to understand and clean the (training) data, which includes ([Google Developers, 2022](https://www.tensorflow.org/tfx/tutorials/tfx/penguin_tfdv)):

1. Analyze the data: Understanding the data types, distributions, and other information (e.g., mean value, or number of uniques) about each feature
1. Define schema: Generating a preliminary schema that describes the data (float; integer, ...)
1. Anomaly detection: Identifying anomalies and missing values in the data with respect to given schema

We analyze the training data to understand important predictor characteristics such as their individual distributions, the degree of missingness within each predictor, potentially unusual values within predictors, relationships between predictors, and the relationship between each predictor and the response and so on {cite:p}`Kuhn2019`. In particular, exploratory data analysis (EDA) is used to understand if there are any challenges associated with the data that can be discovered prior to modeling (like multicollinearity). Furthermore, good visualisations will show you things that you did not expect, or raise new questions about the data {cite:p}`Wickham2016`: A good visualisation might also hint that you’re asking the wrong question, or you need to collect different data. 


### Feature engineering

The understanding gained from our data analysis is now used (in combination with domain knowledge) for feature engineering. The goal of this process is to make it easier to accurately model the data. This process may include encoding of categorical predictors and the creation of new features by simply using the ratio of two predictors or with the help of more complex methodologies like pricipal component analysis.

Typically we als need to handle continuous predictors because predictors may {cite:p}`Kuhn2019`:

- be on vastly different scales.
- follow a skewed distribution where a small proportion of samples are orders of magnitude larger than the majority of the data (i.e., skewness).
- contain a small number of extreme values.
- be censored on the low and/or high end of the range.
- have a complex relationship with the response and is truly predictive but cannot be adequately represented with a simple function or extracted by sophisticated models.
- contain relevant and overly redundant information. That is, the information collected could be more effectively and efficiently represented with a smaller, consolidated number of new predictors while still preserving or enhancing the new predictors’ relationship with the response.


## Model

Once our features have been encoded in a format ready for a modeling algorithm, they can be used in the training of the model. Note that the process of analysis, feature engineering and modeling often requires multiple iterations. The general phases are {cite:p}`Kuhn2021`:

![](https://www.tmwr.org/premade/modeling-process.svg)

The colored segments within the circles signify the repeated data splitting used during resampling. We discuss the model building process in the following sections.

### Select algorithm

For some datasets the best algorithm could be a linear model, while for other datasets it is a neural network. There is no model that is a priori guaranteed to work better. This fact is known as the "No Free Lunch (NFL) theorem" {cite:p}`Wolperet1996`.

The only way to know for sure which model is best is to evaluate them all. Since this is not possible, in practice you make some reasonable assumptions about the data and evaluate only a few reasonable models {cite:p}`Geron2019`. For example, for simple tasks you may evaluate linear models with various levels of regularization, and for a complex problem you may evaluate various neural networks.

### Model training, tuning and evaluation

In the first phase of the model building process, a variety of initial models are generated and their performance is compared during model evaluation. In our model evaluation, we assess the model’s performance metrics, examine residual plots, and conduct other EDA-like analyses to understand how well the models work.

Our first goal in this model building process is to shortlist a few (two to five) promising models {cite:p}`Geron2019`. Let’s assume that we now have a shortlist of promising models. We now need to fine-tune them (hyperparameter-tuning) and/or try to combine the models that perform best. The group (or “ensemble”) will often perform better than the best individual model (just like Random Forests perform better than the individual Decision Trees they rely on), especially if the individual models make very different types of errors. We will cover this topic in more detail in


## Model evaluation

Next, we need to assess the model’s performance metrics, examine residual plots, and conduct other EDA-like analyses to understand how well the models work.  We repeat training and tuning 





XX OLD





- Exploratory data analysis (EDA): Initially there is a back and forth between numerical analysis and visualization of the data  where different discoveries lead to more questions and data analysis “side-quests” to gain more understanding.

- Feature engineering: The understanding gained from EDA results in the creation of specific model terms that make it easier to accurately model the observed data. This can include complex methodologies (e.g., PCA) or simpler features (using the ratio of two predictors). 

- Model tuning and selection (circles with blue and yellow segments): A variety of models are generated and their performance is compared. Some models require parameter tuning where some structural parameters are required to be specified or optimized. The colored segments within the circles signify the repeated data splitting used during resampling.

- Model evaluation: During this phase of model development, we assess the model’s performance metrics, examine residual plots, and conduct other EDA-like analyses to understand how well the models work. 

The last step is **communication**, an absolutely critical part of any data analysis project. It doesn't matter how well your models and visualisation have led you to understand the data unless you can also communicate your results to others.

Surrounding all the data science steps covered above is **programming**. Programming is a cross-cutting tool that you use in every part of the project. You don’t need to be an expert programmer to be a data scientist, but learning more about programming pays off because becoming a better programmer allows you to automate
common tasks, and solve new problems with greater ease.


{cite:p}`Wickham2016`
