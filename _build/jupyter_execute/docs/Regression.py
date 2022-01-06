#!/usr/bin/env python
# coding: utf-8

# # Regression

# One of the simplest regression problems is fitting a line to data, which we saw above — but Scikit-learn also contains more sophisticated regression algorithms.
# Let's start with a line, and create some simple (random) data.

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

X = np.random.random(size=(20, 1))
y = 3 * X.squeeze() + 2 + np.random.randn(20)
plt.plot(X.squeeze(), y, 'o');


# Just as we did in earlier chapter, we will fit the model to the random points we generated on the previous page using LinearRegression estimator, and plot the resulting line.

# In[2]:


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, y)
X_fit = np.linspace(0, 1, 100)[:, np.newaxis]
y_fit = model.predict(X_fit)
plt.plot(X.squeeze(), y, 'o')
plt.plot(X_fit.squeeze(), y_fit);


# In[3]:


from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)
X_fit = np.linspace(0, 1, 100)[:, np.newaxis]
y_fit = model.predict(X_fit)
plt.plot(X.squeeze(), y, 'o')
plt.plot(X_fit.squeeze(), y_fit);


# As a quick exercise, you can explore the RandomForestRegressor object using Tinkerstellar's help features: run code on this page, and you will see the full documentation for RandomForestRegressor class. What arguments are available to RandomForestRegressor? How does the above plot change if you change these arguments?
# These class-level arguments are known as hyperparameters, and we will discuss later how you to select hyperparameters in the model validation section.

# In[4]:


get_ipython().run_line_magic('pinfo', 'RandomForestRegressor')


# In[ ]:




