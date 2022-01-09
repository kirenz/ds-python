# Model

Once our features have been encoded in a format ready for a modeling algorithm, they can be used in the training of the model. Note that the process of analysis, feature engineering and modeling often requires multiple iterations. The general phases are {cite:p}`Kuhn2021`:

![](https://www.tmwr.org/premade/modeling-process.svg)

The colored segments within the circles signify the repeated data splitting used during resampling. We discuss the model building process in the following sections.

## Select algorithm

For some datasets the best algorithm could be a linear model, while for other datasets it is a neural network. There is no model that is a priori guaranteed to work better. This fact is known as the "No Free Lunch (NFL) theorem" {cite:p}`Wolperet1996`.

The only way to know for sure which model is best is to evaluate them all. Since this is not possible, in practice you make some reasonable assumptions about the data and evaluate only a few reasonable models {cite:p}`Geron2019`. For example, for simple tasks you may evaluate linear models with various levels of regularization, and for a complex problem you may evaluate various neural networks.

## Model training, tuning and evaluation

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
