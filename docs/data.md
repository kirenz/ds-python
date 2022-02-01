# Data

There are various options how to manage your data but we won't go into the details of data engineering in this tutorial. However, if you want to learn more about topics like:

- the basics of big data (Hadoop ecosystem and Spark),
- relational and NoSQL databases,
- how to set up a PostgreSQL and MySQL database,
- examples of different data architectures and
- components of machine learning operations (MLOps),

 review this online book: 

 ```{admonition} Data engineering
:class: tip

- [Introduction to Data Engineering](https://kirenz.github.io/data-engineering/docs/intro.html)
```

## Data ingestion

The first step is to ingest the data. This means that you take data stored in a file, a relational database, a NoSQL database or data lakehouse and load it into Python. In our examples, we often use pandas to import CSV files and store it as `df` (short for DataFrame). Next, we get a first impression of the data structure, perform some initial data error corrections and prepare our data for following steps. The process looks like follows:

1. [Import data with pandas](https://kirenz.github.io/pandas/pandas-intro-short.html#read-and-write-data) and store it as df: `df = pd.read_csv(...)`
1. Call `df` to take a look at the 5 top and bottom observations of your data
1. Use `df.info()` to get a quick description of the data, in particular the total number of rows, each attribute’s type, and the number of nonnull values.
1. Perform data corrections to take care of data problems (e.g. wrong data types)
1. Prepare lists of variables for later steps

### Data corrections

Despite the fact that it would be easiest to preprocess your data right away in pandas, we only take care of the most problematic errors (like the occurence of strings in data columns or wrong data formats). We only perform absolutely necessary data preprocessing because processing your data in pandas before passing it to modules like scikit-learn might be problematic for one of the following reasons ([scikit learn developers](https://scikit-learn.org/stable/modules/compose.html#columntransformer-for-heterogeneous-data)):

- Incorporating statistics from data which later becomes our test data into the preprocessors makes cross-validation scores unreliable (known as data leakage), for example in the case of scalers (z transformation) or imputing missing values.

- You may want to include the parameters of the preprocessors in a parameter search (for hyperparameter tuning).

Later we will see that scikit-learn's [ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html#sklearn.compose.ColumnTransformer) helps performing different transformations for different columns of the data  within a **data preprocessing pipeline** that is safe from data leakage and that can be parametrized. To each column, a different transformation can be applied, such as preprocessing or a specific feature extraction method.

As a general rule, we only take care of data errors which can be fixed without the risk of data leakage and which we don't want to include as data preprocessing steps in a pipeline. 

### Variable lists

We often need specific variables for exploratory data analysis as well as data preprocessing steps within a pipeline. We can use pandas functions to create specific lists (provided all columns are stored in the correct data format):

```python

# list of numerical data
list_num = df.select_dtypes(include=[np.number]).columns.tolist()

# list of categorical data
list_cat = df.select_dtypes(include=[object]).columns.tolist()

```

Furthermore, we prepare lists of variables for the following process of data splitting. Note that we use `foo` as placeholder for your outcome variable:

```python

# define outcome variable
y_label = 'foo'

# Select all variables except your y label
X = df.drop(columns=[y_label])

# Create outcome
y = df[y_label]
```

## Data splitting

Before you start analyzing your data, it is a good idea to split your data into a *training* and *test set* {cite:p}`Geron2019`. We do this because this is the only way to know how well a model will generalize to new cases. Furthermore, we will perform exploratory data analysis only on the training data so we don't use insights from the test data during the model building process.

### Training, evaluation and test set

The error rate on new cases is called the *generalization error* (or out-of-sample error), and by evaluating our model on the test set, we get an estimate of this error. This value tells you how well your model will perform on instances it has never seen before. If the training error is low (i.e., your model makes few mistakes on the training set) but the generalization error is high, it means that your model is **overfitting** the training data.

Note that if we want to evaluate different settings (“hyperparameters”) for models, such as the alpha in Lasso, there is still a risk of overfitting on the test set because the parameters can be tweaked until the model performs optimally ([skicit learn developers](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation)). This way, knowledge about the test set can “leak” into the model and evaluation metrics no longer report on generalization performance. To solve this problem, yet another part of the dataset can be held out as a so-called “**validation set**”: training proceeds on the *training set*, after which evaluation is done on the *validation set*, and when the experiment seems to be successful, final evaluation can be done on the *test set*.

However, by partitioning the available data into three sets, we drastically reduce the number of samples which can be used for learning the model, and the results can depend on a particular random choice for the pair of (train, validation) sets ([skicit learn developers](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation)). A solution to this problem is a procedure called cross-validation (CV for short). A test set should still be held out for final evaluation, but the validation set is no longer needed when doing CV. In the basic approach, called k-fold CV, the training set is split into k smaller sets.

<iframe src="https://docs.google.com/presentation/d/e/2PACX-1vTPAoobEeafrN7WzxPwwKBr4G18Yh3P12ru6b123FakIWspNXe6EJU47nBKjfBqs1S7U-2Jwdhm_RKD/embed?start=false&loop=false&delayms=3000" frameborder="0" width="840" height="520" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>

### Train test split

We typically use scikit-learn's [train test split function](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) to perform data splitting: 


```Python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## Data analysis and preprocessing

The goal of this phase is to understand and preprocess the training data. Therefore, we create a special DataFrame called `df_train` where we combine the training features with the corresponding labels:

```Python
df_train = pd.DataFrame(X_train.copy())
df_train = df_train.join(pd.DataFrame(y_train))
```

First we analyze the training data to understand important predictor characteristics such as {cite:p}`Kuhn2019`:

- central tendency and distribution (for numerical data): `df_train.describe().T`
- levels and uniqueness (for categorical data): `df_train.describe(include="category").T`
- the degree of missingness within each predictor: `print(df.isnull().sum())`  

- potentially unusual values within predictors, 
- the relationship between each predictor and the response, 
- relationships between predictors to detect multicollinearity.





In particular, **exploratory data analysis (EDA)** is used to understand if there are any challenges associated with the data that can be discovered prior to modeling (like multicollinearity). Furthermore, good visualisations will show you things that you did not expect, or raise new questions about the data {cite:p}`Wickham2016`: A good visualisation might also hint that you’re asking the wrong question, or you need to collect different data. 

Soemetimes it is also a good idea to defina a preliminary schema that describes the data (e.g., whether a feature has to be present in all examples, allowed value ranges, and other properties)

Finally,  Identifying anomalies and missing values in the data with respect to given schema.


 ```{admonition} Exploratory data analysis 
:class: tip

- [Data analysis in pandas](https://kirenz.github.io/pandas/intro.html)
- [Data exploration with seaborn](https://seaborn.pydata.org/) 
- [From Data to Viz](https://www.data-to-viz.com/) leads you to the most appropriate graph for your data.
- [Imputation of missing values](https://scikit-learn.org/stable/modules/impute.html)
- [Handling multicollinear features](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html#handling-multicollinear-features)

```

Note that the usage of **data preprocessing pipelines** is considered a best practice to help avoid leaking statistics from your test data into the trained model. To learn more about pipelines, see section [](section:data:pipeline). 

## Feature engineering

> "Applied machine learning is basically feature engineering" Andrew Ng 

The understanding gained from our data analysis is now used for feature engineering, which is the process of using domain knowledge to extract meaningful features (attributes) from raw data. The goal of this process is to create new features which improve the predictions from our model and my include steps like:
 
- Feature extraction (reduce the number of features by combining existing features)
- Feature creation (make new features)
- Feature transformation (transform features)

Features may contain relevant but overly redundant information. That is, the information collected could be more effectively and efficiently represented with a smaller, consolidated number of new predictors while still preserving or enhancing the new predictors’ relationship with the response {cite:p}`Kuhn2019`. In that case, **feature extraction** can be achieved by simply using the ratio of two predictors or with the help of more complex methods like pricipal component analysis. 

Typically, we als need to perform **feature transformationa** like the encoding of categorical features and the transformation of continuous predictors, because predictors may {cite:p}`Kuhn2019`:

- be on vastly different scales.
- follow a skewed distribution where a small proportion of samples are orders of magnitude larger than the majority of the data (i.e., skewness).
- contain a small number of extreme values.
- be censored on the low and/or high end of the range.

 ```{admonition} Feature engineering 
:class: tip

- [Feature extraction in scikit-learn](https://scikit-learn.org/stable/modules/feature_extraction.html)

- Review {cite:t}`Kuhn2019` for a detailed discussion of feature engineering methods.
```

Again, the usage of **pipelines** is considered best practice to help avoid leaking statistics from your test data into the trained model. To learn more about pipelines, review the next section [](section:data:pipeline). 

(section:data:pipeline)=
## Pipelines in scikit-learn

scikit-learn provides a library of transformers for data preprocessing and feature engineering, which may 

- clean data (see [preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing), 
- reduce features (see [unsupervised dimensionality reduction](https://scikit-learn.org/stable/modules/unsupervised_reduction.html#data-reduction)), or 
- extract features (see [feature extraction](https://scikit-learn.org/stable/modules/feature_extraction.htmln)).

Just as it is important to test a model on data held-out from training, data preprocessing (such as standardization, etc.) and similar data transformations similarly should be learnt from a training set and applied to held-out data for prediction. For example, the standardization of numerical features should always be performed after data splitting and only from training data. Furthermore, we obtain all necessary statistics (mean and standard deviation) from training data and use them on test data. Note that we don’t standardize dummy variables (which only have values of 0 or 1):

```python
from sklearn.preprocessing import StandardScaler

numerical_features = ['a', 'b']

scaler = StandardScaler().fit(X_train[numerical_features]) 

X_train[numerical_features] = scaler.transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# ...
```

Instead of performing data preprocessing as shown above, we should use a pipeline. **Pipelines** are a best practice to help avoid leaking statistics from your test data into the trained model (e.g. during cross-validation). Here an example of a data preprocessing pipeline with imputation of missing data and standardization:

```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

numeric_features = ['a', 'b']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_features = ['c', 'd']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# ...
```

Note that we are able to combine data preprocessing with our modeling builing in one pipeline. To learn how to create pipelines, see:

 ```{admonition} Pipelines 
:class: tip

- scikit-learn's [pipelines documentation](https://scikit-learn.org/stable/modules/compose.
html)

- [Regression example with preprocessing pipeline](https://kirenz.github.io/regression/docs/case-duke-sklearn.html)

```