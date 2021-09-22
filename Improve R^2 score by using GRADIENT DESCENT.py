#!/usr/bin/env python
# coding: utf-8

# In[104]:


import numpy as np
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[123]:


df=pd.read_csv("D:\\data\\diabetes.csv")


# In[5]:


df


# In[157]:


df.info()


# In[6]:


df.corr()


# In[106]:


y=df[["Outcome"]]
x=df.iloc[:,[1]]


# In[81]:


x


# In[107]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=2)


# In[108]:


reg=LinearRegression()
reg.fit(x_train,y_train)


# In[109]:


print(reg.coef_)
print(reg.intercept_)


# In[110]:


y_pred=reg.predict(x_test)


# In[86]:


r2_score(y_test,y_pred)


# In[111]:


x_train.shape


# In[112]:


#Applying Gradient Descent for better r2 score


# In[137]:


from sklearn.linear_model import SGDRegressor


# In[139]:


sgd=SGDRegressor(max_iter=100,penalty=None,eta0=0.1,alpha=0.01)


# In[142]:



sgd.fit(x_train,y_train)


# In[143]:


y_pred=sgd.predict(x_test)


# In[144]:


r2_score(y_test,y_pred)


# In[145]:


# so the chosen model is worse


# In[ ]:




