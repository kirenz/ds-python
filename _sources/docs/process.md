# Data Science Lifecycle

 From a high-level perspective, a typical data science project looks something like this :

```{image} ../_static/img/lifecycle-data.png
:alt: datascience
:class: bg-primary mb-1
:width: 600px
:align: center
```

<br>

## Plan

First, you have to define a plan of what you want to achive with your data science project. To do this, we start with the business model, which describes the rationale of how your organization creates, delivers and captures value. The complete process looks something like this:

:::{note}

1. Identify use case: Use the business model canvas.
2. Frame the problem: Provide a statement of what is to be learned and how decisions should be made.
3. Identify variables or labels: for structured data problems, we need to identify potentially relevant variables; for unstructured data problems, we need to define labels.
4. Define success metrics: Write down your metrics for success and failure with the data science project. 
:::

To learn more about the data science planning phase, review this presentation: 

<iframe src="https://docs.google.com/presentation/d/e/2PACX-1vR3mAfcepfacMwk7_ob-uPjSX6aMLISTxC2C1DEOyMS5HdO1RSY8fSbBdPP21JjKP0fHKoE46719xjJ/embed?start=false&loop=false&delayms=3000" frameborder="0" width="960" height="569" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>

## Data

Typically, you first have to import your data. This means that you take data stored in a file, a relational database, NO-SQL database, data lakehouse or web API, and load it into Python. 

If you want to learn more about the basics of data engineering, review this resource: 

- Big data (Hadoop ecosystem, )
- Databases
- Architectures
- Machine Learning Operations MLOPS

- [Introduction to Data Engineering](https://kirenz.github.io/data-engineering/docs/intro.html)



:::{note}
In our examples, we usually use [pandas to import CSV files](https://kirenz.github.io/pandas/pandas-intro-short.html#read-and-write-data)
:::



Once you’ve imported your data, it is a good idea to tidy it. **Tidying** your data means storing it in a consistent form that matches the semantics of the dataset with the way it is stored (we usually work with tabular data in the form of excel spreadsheets). In brief, when your data is tidy, each column is a variable, and each row is an observation. Tidy data is important because the consistent structure lets you focus your struggle on questions about the data.

Once you have tidy data, a common first step is to transform it. **Transformation** includes narrowing in on observations of interest (like all people in one city, or all data from the last year), creating new variables that are functions of existing variables (like computing velocity from speed and time), and calculating a set of summary statistics (like counts or means). Together, tidying and transforming are called *wrangling*, because getting your data in a form that’s natural to work with often feels like a fight! Once you have tidy data with the variables you need, there are two main engines of knowledge generation: visualisation and modelling. These have complementary strengths and weaknesses so any real analysis will iterate between them many times. 

**Visualisation** is a fundamentally human activity. A good visualisation will show you things that you did not expect, or raise new questions about the data. A good visualisation might also hint that you’re asking the wrong question, or you need to collect different data. Visualisations can surprise you, but don’t scale particularly well because they require a human to interpret them.

**Models** are complementary tools to visualisation. Once you have made your questions sufficiently precise, you can use a model to answer them. 

Note that the cycle of analysis, modeling, and visualization often requires multiple iterations. This iterative process is especially true for modeling. The general phases are {cite:p}`Kuhn2021`:

![](https://www.tmwr.org/premade/modeling-process.svg)


- Exploratory data analysis (EDA): Initially there is a back and forth between numerical analysis and visualization of the data  where different discoveries lead to more questions and data analysis “side-quests” to gain more understanding.

- Feature engineering: The understanding gained from EDA results in the creation of specific model terms that make it easier to accurately model the observed data. This can include complex methodologies (e.g., PCA) or simpler features (using the ratio of two predictors). 

- Model tuning and selection (circles with blue and yellow segments): A variety of models are generated and their performance is compared. Some models require parameter tuning where some structural parameters are required to be specified or optimized. The colored segments within the circles signify the repeated data splitting used during resampling.

- Model evaluation: During this phase of model development, we assess the model’s performance metrics, examine residual plots, and conduct other EDA-like analyses to understand how well the models work. 

The last step is **communication**, an absolutely critical part of any data analysis project. It doesn't matter how well your models and visualisation have led you to understand the data unless you can also communicate your results to others.

Surrounding all the data science steps covered above is **programming**. Programming is a cross-cutting tool that you use in every part of the project. You don’t need to be an expert programmer to be a data scientist, but learning more about programming pays off because becoming a better programmer allows you to automate
common tasks, and solve new problems with greater ease.


{cite:p}`Wickham2016`