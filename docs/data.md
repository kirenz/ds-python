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

### Import data

The first step is to import the data. This means that you take data stored in a file, a relational database, a NoSQL database or data lakehouse and load it into Python. In our examples, we often use [pandas to import CSV files](https://kirenz.github.io/pandas/pandas-intro-short.html#read-and-write-data) and store it as `df` (short for DataFrame): 

```Python
path_to_file = "my-file.csv"

df = pd.read_csv(path_to_file)
```

### Data structure

Next, we get a first impression of the data structure: 

- Take a look at the 5 top and bottom observations of your data:

```Python
df
```

- Print the number of observations and columns:

```Python
print(f"We have {len(df.index):,} observations and {len(df.columns)} columns in our dataset.")
```

- View a description of the data, in particular the total number of rows, each attribute’s type, and the number of nonnull values:

```Python
df.info()
```

### Data corrections

Despite the fact that it would be easiest to preprocess your data right away in pandas, we only take care of the most problematic errors (like the occurence of strings in data columns or wrong data formats). We only perform absolutely necessary data preprocessing because processing your data in pandas before passing it to modules like scikit-learn might be problematic for one of the following reasons ([scikit learn developers](https://scikit-learn.org/stable/modules/compose.html#columntransformer-for-heterogeneous-data)):

- Incorporating statistics from data which later becomes our test data into the preprocessors makes cross-validation scores unreliable (known as data leakage), for example in the case of scalers (z transformation) or imputing missing values.

- You may want to include the parameters of the preprocessors in a parameter search (for hyperparameter tuning).

Later we will see that scikit-learn's [ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html#sklearn.compose.ColumnTransformer) helps performing different transformations for different columns of the data  within a **data preprocessing pipeline** that is safe from data leakage and that can be parametrized. To each column, a different transformation can be applied, such as preprocessing or a specific feature extraction method.

As a general rule, we only take care of data errors which can be fixed without the risk of data leakage and which we don't want to include as data preprocessing steps in a pipeline. 

Example of data type transformation of one variable:

```Python
df['foo'] = df['foo'].astype('float')
```

Example of data type transformation for multiple variables:

```Python
# Convert to categorical

cat_convert = ['foo1', 'foo2', 'foo3']

for i in cat_convert:
    df[i] = df[i].astype("category")
```

### Creation of new variables

During the creation of your [plan](plan.md), you maybe gained knowledge about possible ways to derive new variables from already existing columns in your dataset (e.g. through simple variable combinations). If this is the case, now would be a good time to create these variables. 

Pandas offers multiple ways to derive new columns from existing columns (see this [pandas tutorial](https://pandas.pydata.org/docs/getting_started/intro_tutorials/05_add_columns.html) for more examples). Note that you create a new column by assigning the output to the DataFrame with a new column name in between the [] and that operations are element-wise (i.e., no need to loop over rows):

```Python
df[my_new_feature] = df[feature_1] / df[feature_2]

df[my_newest_feature] = (df[feature_1] + df[feature_2])/2
```

(section:data:variable-lists)=
### Variable lists

We often need specific variables for exploratory data analysis as well as data preprocessing steps. We can use pandas functions to create specific lists (provided all columns are stored in the correct data format):

```python
# list of numerical data
list_num = df.select_dtypes(include=[np.number]).columns.tolist()

# list of categorical data
list_cat = df.select_dtypes(include=['category']).columns.tolist()
```

Furthermore, we prepare lists of variables for the following process of data splitting. Note that we use `foo` as placeholder for our outcome variable:

```python
# define outcome variable as y_label
y_label = 'foo'

# Select features
features = df.drop(columns=[y_label]).columns.tolist()
X = df[features]

# Create response
y = df[y_label]
```

## Data splitting

Before you start analyzing your data, it is a good idea to split your data into a *training* and *test set* {cite:p}`Geron2019`. We do this because this is the only way to know how well a model will generalize to new cases. Furthermore, we will perform exploratory data analysis only on the training data so we don't use insights from the test data during the model building process.

### Training, evaluation and test set

The error rate on new cases is called the *generalization error* (or out-of-sample error), and by evaluating our model on the test set, we get an estimate of this error. This value tells you how well your model will perform on instances it has never seen before. If the training error is low (i.e., your model makes few mistakes on the training set) but the generalization error is high, it means that your model is **overfitting** the training data.

Note that if we want to evaluate different settings (“hyperparameters”) for models, such as the alpha in Lasso, there is still a risk of overfitting on the test set because the parameters can be tweaked until the model performs optimally ([skicit learn developers](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation)). This way, knowledge about the test set can “leak” into the model and evaluation metrics no longer report on generalization performance. To solve this problem, yet another part of the dataset can be held out as a so-called “**validation set**”: training proceeds on the *training set*, after which evaluation is done on the *validation set*, and when the experiment seems to be successful, final evaluation can be done on the *test set*.

However, by partitioning the available data into three sets, we drastically reduce the number of samples which can be used for learning the model, and the results can depend on a particular random choice for the pair of (train, validation) sets ([skicit learn developers](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation)). A solution to this problem is a procedure called cross-validation (CV for short). A test set should still be held out for final evaluation, but the validation set is no longer needed when doing CV. In the basic approach, called k-fold CV, the training set is split into k smaller sets.

<iframe src="https://docs.google.com/presentation/d/e/2PACX-1vTPAoobEeafrN7WzxPwwKBr4G18Yh3P12ru6b123FakIWspNXe6EJU47nBKjfBqs1S7U-2Jwdhm_RKD/embed?start=false&loop=false&delayms=3000" frameborder="0" width="840" height="520" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>

### Train and test split

We typically use scikit-learn's [train test split function](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) to perform data splitting and use `random_state` to make this notebook's output identical at every run (we arbitrarily set the number to 42 but you can choose any other number):

```Python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Data exploration set

Furthermore, we create a new DataFrame called `df_train` where we combine the training features with the corresponding y labels. We will use this data for our exploratory data analysis.

```Python
df_train = pd.DataFrame(X_train.copy())
df_train = df_train.join(pd.DataFrame(y_train))
```

(section:data:analyze)=
## Aanalyze data

The goal of this phase is to understand the training data. In particular, exploratory data analysis (EDA) is used to understand if there are any challenges associated with the data that can be discovered prior to modeling (like potentially unusual values within predictors or multicollinearity between predictors).  

Furthermore, good visualisations will show you things that you did not expect, or raise new questions about the data {cite:p}`Wickham2016`: A good visualisation might also hint that you’re asking the wrong question, or you need to collect different data.  

 ```{admonition} Exploratory data analysis  
:class: tip
- [Data analysis in pandas](https://kirenz.github.io/pandas/intro.html)
- [Data exploration with seaborn](https://seaborn.pydata.org/) 
- [From Data to Viz](https://www.data-to-viz.com/) leads you to the most appropriate graph for your data.
```

Next, we analyze the training data to understand important predictor characteristics {cite:p}`Kuhn2019`:  

:::{Note}
We use some lists created in [](section:data:variable-lists) for some of the steps shown below
:::  

### Numerical data

- Numerical data: central tendency and distribution:

```Python
# summary of numerical attributes
df_train.describe().round(2).T
```

```Python
# histograms
df_train.hist(figsize=(20, 15));
```

### Categorical data

- Categorical data: levels and uniqueness:

```Python
df_train.describe(include="category").T 
```  

```Python
for i in list_cat:
    print(i, "\n", df_train[i].value_counts())
```  

```Python
for i in list_cat:
    print(df_train[i].value_counts().plot(kind='barh', title=i));
```  

- Numerical data grouped by categorical data:

```Python
# median
for i in list_cat:
    print(df_train.groupby(i).median().round(2).T)
```

```Python
# mean
for i in list_cat:
    print(df_train.groupby(i).mean().round(2).T)
```

```Python
# standard deviation
for i in list_cat:
    print(df_train.groupby(i).std().round(2).T)
```

### Relationships

#### Correlation with response

- Relationship between each predictor and the response:

```Python
sns.pairplot(data=df_train, y_vars=y_label, x_vars=features);
```

```Python
# pairplot with one categorical variable
sns.pairplot(data=df_train, y_vars=y_label, x_vars=features, hue="a_categorical_variable");
```

```Python
# inspect correlation
corr = df_train.corr()
corr_matrix[y_label].sort_values(ascending=False)

```

#### Correlation between predictors

- Relationships between predictors to detect multicollinearity.

```Python
sns.pairplot(df_train);
```

```Python
# inspect correlation
corr = df_train.corr()
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(220, 10, as_cmap=True)

fig, ax = plt.subplots(figsize=(10,10)) 
sns.heatmap(corr, mask=mask, cmap=cmap, annot=True,  
            square=True, annot_kws={"size": 12});
```

Instead of inspecting the correlation matrix, a better way to assess multicollinearity is to compute the variance inflation factor (VIF). The smallest possible value for VIF is 1, which indicates the complete absence of collinearity. Typically in practice there is a small amount of collinearity among the predictors. As a rule of thumb, a VIF value that exceeds 5 or 10 indicates a problematic amount of collinearity.

```Python
# calculate variance inflation factor

# create new dataframe X_ and add a constant
X_ = df_train[list_num]
X_ = X_.drop(columns=y_label)
X_ = add_constant(X_)

# For each X, calculate VIF and save in dataframe
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X_.values, i) for i in range(X_.shape[1])]
vif["Feature"] = X_.columns

vif.round(2)
```

## Define schema

Usually it is a good idea to define some sort of schema that describes the expected properties of the data. Some of these properties are ([TensorFlow](https://www.tensorflow.org/tfx/data_validation/get_started)):

- which features are expected to be present
- their type
- the number of values for a feature in each example
- the presence of each feature across all examples
- the expected domains of features.

We don't cover this topic in detail here but if you want to learn more about schemas, check out the following resources:

- [The SchemaGen TFX Pipeline Component](https://www.tensorflow.org/tfx/guide/schemagen)
- [List of schema anomalies provided by TFX](https://github.com/tensorflow/data-validation/blob/master/g3doc/anomalies.md)
- [Simple solution for pandas](https://stackoverflow.com/questions/54971410/how-do-you-specify-a-pandas-dataframe-schema-structure-in-a-docstring/61041468#61041468)
- [3rd party tool PandasSchema](https://multimeric.github.io/PandasSchema/)

## Anomaly detection

Next, we need to identify missing values and anomalies in the data (with respect to a given schema).  

Note that we just gain insights and don't perform any data preprocessing during the phase of anomaly detection. We only need to decide how to deal with the issues we detect. All data transformations will be performed during feature engineering.  

### Missing values

We check the degree of missingness within each predictor in the original dataframe to avoid code duplication (otherwise we first would perform all checks on `df_train` and afterwards on `df_test`).

:::{Note}
We use the original dataframe `df` to check for missing values
:::

```Python
# missing values will be displayed in yellow
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis');
```

```Python
# absolute number of missing values
print(df.isnull().sum())
```

```Python
# percentage of missing values
df.isnull().sum() * 100 / len(df)
```

If we find missing cases in our data, we need to decide how to deal with them. For example, we could:

1. Get rid of the corresponding observations.
1. Get rid of the whole attribute.
1. Set the values to some value (zero, the mean, the median, etc.).

We cover this topic in more detail in the section about [fixing missing values](section:data:fix-missing).

### Outlier and novelty detection

Many applications require being able to decide whether a new observation belongs to the same distribution as existing observations (it is an inlier), or should be considered as different (it is an outlier). Two important distinctions must be made ([scikit-learn developers](https://scikit-learn.org/stable/modules/outlier_detection.html)):

- **outlier detection**: The training data contains outliers which are defined as observations that are far from the others. Outlier detection estimators thus try to fit the regions where the training data is the most concentrated, ignoring the deviant observations.

- **novelty detection**: The training data is not polluted by outliers and we are interested in detecting whether a new observation is an outlier. In this context an outlier is also called a novelty.

There are various strategies to deal with unusual cases which we will cover in the section [data preprocessing](section:data:fix-outliers).

## Feature engineering

> "Applied machine learning is basically feature engineering" Andrew Ng 

The understanding gained in [data analysis](section:data:analyze) is now used for data preprocessing and feature engineering, which is the process of using domain knowledge to extract meaningful features (attributes) from raw data. The goal of this process is to clean our data and to create new features which improve the predictions from our model and my include steps like:
 
- Data preprocessing (encode categorical data, fix missing values and outliers) 
- Feature extraction (reduce the number of features by combining existing features)
- Feature creation (make new features)
- Feature transformation (transform features)

Note that the usage of **data pipelines** is considered best practice to help avoid leaking statistics from your test data into the trained model during data preprocessing and feature engineering. Therefore, we first take a look at pipelines.

(section:data:pipeline)=
### Pipelines

Just as it is important to test a model on data held-out from training, data preprocessing (such as standardization, etc.) and data transformations similarly should be learnt from a training set and applied to held-out data for prediction. For example, the standardization of numerical features should always be performed after data splitting and only from training data. 

scikit-learn provides a library of transformers for data preprocessing and feature engineering. Note that we are able to combine data preprocessing with our modeling builing in one pipeline. 

 ```{admonition} Pipelines 
:class: tip
- scikit-learn's [pipelines documentation](https://scikit-learn.org/stable/modules/compose.
html)

```

### Data preprocessing

<!--
Furthermore, we obtain all necessary statistics (mean and standard deviation) from training data and use them on test data. Note that we don’t standardize dummy variables (which only have values of 0 or 1):

 Here an example of a data preprocessing pipeline with imputation of missing data and standardization:

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
```

(section:data:categorical)=
#### Encode categorical data

```Python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

categorical_features = ['c', 'd']

categorical_transformer = OneHotEncoder(handle_unknown='ignore')

```
-->

(section:data:fix-missing)=
#### Fix missing values

Here an example of a data preprocessing pipeline to fix missing values: 

A) We start with the imputation of missing data for numerical data:

```Python
from sklearn.pipeline import Pipeline

numeric_features = ['a', 'b']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))]
```

B) Next, we impute missing values in categorical values:

```Python
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing'))]
```

C) Now we combine the two with each other:

```Python
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

data_prepared = preprocessor.fit_transform(housing)        
```



(section:data:fix-outliers)=
#### Fix outliers

One efficient way of performing outlier detection in high-dimensional datasets is to use the random forest algorithm [IsolationForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html#sklearn.ensemble.IsolationForest). When we perform fit on our variable, it returns labels for it: -1 for outliers and 1 for inliers.


```Python
from sklearn.ensemble import IsolationForest

list_detect = df_train.drop(y_label).columns.tolist()

clf = IsolationForest(random_state=42)

clf.fit(X_train[list_detect])
y_pred_train = clf.predict(X_train[list_detect])
y_pred_test = clf.predict(X_test[list_detect])

```

- [Imputation of missing values](https://scikit-learn.org/stable/modules/impute.html)
- [Handling multicollinear features](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html#handling-multicollinear-features)

### Feature extraction

For example, features may contain relevant but overly redundant information. That is, the information collected could be more effectively and efficiently represented with a smaller, consolidated number of new predictors while still preserving or enhancing the new predictors’ relationship with the response {cite:p}`Kuhn2019`. In that case, **feature extraction** can be achieved by simply using the ratio of two predictors or with the help of more complex methods like pricipal component analysis. 

- reduce features (see [unsupervised dimensionality reduction](https://scikit-learn.org/stable/modules/unsupervised_reduction.html#data-reduction)), or 

### Feature creation

- extract features (see [feature extraction](https://scikit-learn.org/stable/modules/feature_extraction.htmln)).

### Feature transformation

Typically, we als need to perform feature transformations like the encoding of categorical features and the transformation of continuous predictors, because predictors may {cite:p}`Kuhn2019`:

- be on vastly different scales.
- follow a skewed distribution where a small proportion of samples are orders of magnitude larger than the majority of the data (i.e., skewness).
- contain a small number of extreme values.
- be censored on the low and/or high end of the range.

 ```{admonition} Feature engineering 
:class: tip

- [Feature extraction in scikit-learn](https://scikit-learn.org/stable/modules/feature_extraction.html)

- Review {cite:t}`Kuhn2019` for a detailed discussion of feature engineering methods.
```