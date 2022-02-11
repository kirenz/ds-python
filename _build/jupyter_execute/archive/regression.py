#!/usr/bin/env python
# coding: utf-8

# # Regression in scikit-learn

# This tutorial is based on content from the excellent app [Tinkerstellar](https://tinkerstellar.com).
# 

# ## Data
# 
# To demonstrate some regression algorithms, we first create some simple data. Note that the squeeze() function simply removes one-dimensional entry from the shape of the given array.

# In[1]:


# HIDE CODE
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# create 20 random points
X = np.random.rand(20, 1)
# create y as a function from x and add some noise
y = 3 * X.squeeze() + 2 + np.random.randn(20)
# make a plot
plt.plot(X, y, 'o');


# ## Linear regression

# We will fit the model to the random points we generated using LinearRegression estimator, and plot the resulting line.

# In[2]:


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, y)

X_fit = np.linspace(0, 1, 100)[:, np.newaxis]
y_fit = model.predict(X_fit)


# In[3]:


# HIDE CODE
plt.plot(X, y, 'o')
plt.plot(X_fit, y_fit);


# ## Random Forest

# In[4]:


from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)

X_fit = np.linspace(0, 1, 100)[:, np.newaxis]
y_fit = model.predict(X_fit)


# In[5]:


plt.plot(X.squeeze(), y, 'o')
plt.plot(X_fit.squeeze(), y_fit);

