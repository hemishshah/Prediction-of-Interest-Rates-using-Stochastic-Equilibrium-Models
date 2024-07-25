#!/usr/bin/env python
# coding: utf-8

# In[13]:


import sys
sys.path.append("..")


# In[14]:


import pandas as pd
from ratesy.ratesData import ratesData
from ratesy.models import Vasiceck, Normal,Merton
import matplotlib.pyplot as plt


# In[15]:


# Read data 
df = pd.read_excel(r"Interest rates Data.xlsx",sheet_name = 'Sheet1')
rd = ratesData(df,starting_column_name="cmt",conversion=1/100)
print(df.head())


# # Vasiceck  Model Fitting

# ## Fit the model

# In[16]:


vs = Vasiceck(rd)
result = vs.optimize_func()


# ## Get the model intrest rates from fitted data

# In[17]:


df_vasi = vs.step().data
print('The predicted interest rates using Vasicek Model: \n', df_vasi.head())


# # Merton Model 

# ## Fit the Merton Model

# In[18]:


mr = Merton(rd)
result = mr.optimize_func()


# ## Get the model interest rates

# In[19]:


df_merton = mr.step().data
print('The predicted interest rates using Merton Model: \n',df_merton.head())


# # Normal Model

# ## Fit the merton model

# In[20]:


nr = Normal(rd)
result = nr.optimize_func()


# # Get the model Intrest rates

# In[21]:


df_normal = nr.step().data
print('The predicted interest rates using Normal Model: \n', df_normal.head())


# In[22]:


df_actual = rd.data


# In[23]:


fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(df_vasi.index,df_vasi['par_2.0'], label="Vascicek")
ax.plot(df_merton.index,df_merton['par_2.0'], label="Merton")
ax.plot(df_normal.index,df_normal['par_2.0'], label="Normal")
ax.plot(df_actual.index,df_actual['cmt2.0'], label="actual")
ax.set_title("Comparision of fitted 2 year Intrest Rates against the actual Inrest Rates ")
ax.legend()


# In[ ]:
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(df_vasi.index,df_vasi['par_5.0'], label="Vascicek")
ax.set_title("Predicted Vasicek model rates")
ax.legend()




# In[ ]:
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(df_actual.index,df_actual['cmt5.0'], label="actual")
ax.set_title("The actual Interest Rates ")
ax.legend()


fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(df_vasi.index,df_merton['par_5.0'], label="Merton")
ax.set_title("Predicted Merton model rates")
ax.legend()

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(df_vasi.index,df_normal['par_5.0'], label="Normal")
ax.set_title("Predicted Normal model rates")
ax.legend()

#--------------------------------
import numpy as np

actual = (df_actual.to_numpy()[:,1:].flatten())
vasicek_rmse = df_vasi.to_numpy().flatten()
merton_rmse = (df_merton.to_numpy().flatten())
normal_rmse = (df_normal.to_numpy().flatten())

    
mse1 = np.mean((actual - vasicek_rmse)**2)
rmse1 = np.sqrt(mse1)

mse2 = np.mean((actual - merton_rmse)**2)
rmse2 = np.sqrt(mse2)

mse3 = np.mean((actual - normal_rmse)**2)
rmse3 = np.sqrt(mse3)



