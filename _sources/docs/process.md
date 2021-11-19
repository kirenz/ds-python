# Programming process

 A typical data science project looks something like this {cite:p}`Wickham2016`:

## Import data

First you must **import** your data into Python. This typically means that you take data stored in a file, cloud database, or web API, and load it into Python.

Once you’ve imported your data, it is a good idea to tidy it.
**Tidying** your data means storing it in a consistent form that matches the semantics of the dataset with
the way it is stored. In brief, when your data is tidy, each column is a variable, and each row is an observation. Tidy data is important because the consistent structure lets you focus your struggle on questions about the data.

Once you have tidy data, a common first step is to transform it. **Transformation** includes narrowing in on observations of interest (like all people in one city, or all data from the last year), creating new variables that are functions of existing variables (like computing velocity from speed and time), and calculating a set of summary statistics (like counts or means). Together, tidying and transforming are called *wrangling*, because getting your data in a form that’s natural to work with often feels like a fight!

Once you have tidy data with the variables you need, there are two main engines of knowledge generation: visualisation and modelling. These have complementary strengths and weaknesses so any real analysis will iterate between them many times. **Visualisation** is a fundamentally human activity. A good visualisation will show you things that you did not expect, or raise new questions about the data. A good visualisation might also hint that you’re asking the wrong question, or you need to collect different data. Visualisations can surprise you, but don’t scale particularly well because they require a human to interpret them.

**Models** are complementary tools to visualisation. Once you have made your questions sufficiently precise,
you can use a model to answer them. Models are a fundamentally mathematical or computational tool,
 so they generally scale well. 

The last step is **communication**, an absolutely critical part of any data analysis project.
It doesn't matter how well your models and visualisation have led you to understand the data unless
you can also communicate your results to others.


Surrounding all the data science steps covered above is **programming**. Programming is a cross-cutting tool that you use in every part of the project. You don’t need to be an expert programmer to be a data scientist, but learning more about programming pays off because becoming a better programmer allows you to automate
common tasks, and solve new problems with greater ease.