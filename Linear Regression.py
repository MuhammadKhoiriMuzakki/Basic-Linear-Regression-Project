#!/usr/bin/env python
# coding: utf-8

# In[14]:


# Muhammad Khoiri Muzakki - 1703015123
print(__doc__) #Objek Python memiliki atribut yang disebut __doc__ yang menyediakan dokumentasi dari objek tersebut.


# In[15]:


import matplotlib.pyplot as plt #memasukan beberapa fungsi pyplot yang terapat pada library matplotlib
import numpy as np #memasukan library numpy


# In[16]:


from sklearn import datasets, linear_model #memasukan dataset dan model linear yg tersedia dari scikit learn
from sklearn.metrics import mean_squared_error, r2_score #memasukan beberapa fungsi yang tersedia dari scikit learn


# In[17]:


# memuat dataset diabetes
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)


# In[18]:


# Use only one feature
diabetes_X = diabetes_X[:, np.newaxis, 2]


# In[19]:


# membagi data menjadi data training dan testing training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# membagi targets menjadi training dan testing training/testing sets
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]


# In[20]:


# membuat objek linear regression
regr = linear_model.LinearRegression()


# In[21]:


# Train/latih model menggunakan dataset training sets
regr.fit(diabetes_X_train, diabetes_y_train)


# In[22]:


# Membuat prediksi menggunakan dataset testing sets
diabetes_y_pred = regr.predict(diabetes_X_test)


# In[23]:


# Koefisien
print('Coefficients: \n', regr.coef_)
# kesalahan kuadrat rata-rata
print('Mean squared error: %.2f'
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# Koefisien determinasi: 1 adalah prediksi sempurna
print('Coefficient of determination: %.2f'
      % r2_score(diabetes_y_test, diabetes_y_pred))


# In[25]:


# keluaran plot
plt.scatter(diabetes_X_test, diabetes_y_test,  color='red')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show() #menampilkan


# In[ ]:




