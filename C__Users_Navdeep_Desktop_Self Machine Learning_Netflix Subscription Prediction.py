#!/usr/bin/env python
# coding: utf-8

# ## Process
# We will use Time Series Forecasting, we will estimate the expected number of new subscribers for Netflix, in a given time period and better understand the growth potential of their business.

# We will:
#     Step 1: Gather the historical Netflix Subscription data.
#     Step 2: Preprocess and clean the data.
#     Step 3: Explore and analyze the time series patterns.
#     Step 4: Choose the best model for time series forcasting model.
#     Step 5: Split the data and train the model using the training data to avoid             the overfitting.
#     Step 6: Forecast the upcoming Netflix subscription counts.

# In[5]:


#Importing the dependencies

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio
pio.templates.default ='plotly_white'
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


#Importing the data:

df = pd.read_csv(r"C:\Users\Navdeep\Desktop\Self Machine Learning\Netflix Subscription Prediction\Netflix-Subscriptions.csv")


# In[8]:


df.head()


# The dataset contains subscription counts of Netflix at the start of each quarter from 2013 to 2023.

# In[10]:


#Before moving forward,
#let’s convert the Time Period column into a datetime format:

df['Time Period'] = pd.to_datetime(df['Time Period'], format= '%d/%m/%Y')


# In[11]:


df.head()


# In[13]:


#Taking a look at the quarterly subscription growth of Netflix

fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Time Period'],
                        y = df['Subscribers'],
                        mode ='lines', name= 'Subscribers'))
fig.update_layout(title= 'Netflix Quarterly Subcriptions Growth',
                 xaxis_title = 'Date',
                 yaxis_title = 'Netflix Subscriptions')

fig.show()


# Here, we can see that the growth of Netflix subscribers is not seasonal. Now we will use ARIMA model to forecast the new users.

# In[14]:


#Calculate the quarterly growth rate
df['Quarterly Growth Rate'] = df['Subscribers'].pct_change()*100

#Create a new column for bar color
#(green for positive growth, red for negative growth)
df['Bar Color'] = df ['Quarterly Growth Rate'].apply(lambda x: 'green' if x> 0 else 'red')

#Plotting the growth rate

fig = go.Figure()
fig.add_trace(go.Bar(
x = df['Time Period'], 
y = df['Quarterly Growth Rate'],
marker_color = df['Bar Color'],
name = 'Quarterly Growth Rate'))

fig.update_layout(title= 'Netflix Quarterly Subscriptions Growth Rate',
                 xaxis_title = 'Time Period',
                 yaxis_title = 'Quarterly Growth Rate (%)')

fig.show()


# Here, we have created a bar chart using quarterly growth data for all of the Netflix's data. The green part shows the increase in the number of users, and the red part shoes the drop in the number of new subscriptions. 

# In[16]:


#Let's look at the yearly growth rate

df['Year'] = df['Time Period'].dt.year
yearly_growth = df.groupby('Year')['Subscribers'].pct_change().fillna(0)*100

#Creating a new column for bar color
#(Green for positive and red for negative)

df['Bar Color'] = yearly_growth.apply(lambda x: 'green' if x > 0 else 'red')


#Plot the yearly subscriber growth rate using the bar graphs
fig = go.Figure()
fig.add_trace(go.Bar(
x= df['Year'],
y = yearly_growth,
marker_color = df['Bar Color'],
name = 'Yearly Growth Rate'))

fig.update_layout(title ='Netflix Yearly Subscriber Growth Rate',
                 xaxis_title = 'Year',
                 yaxis_title = 'Yearly Growth Rate (%)')

fig.show()


# This bar graph tells us about the continuous growth from 2013 to 2021 and just a bump in the growth rate in 2022.

# ## Using ARIMA for Forecasting Netflix Quarterly Subcriptions
# 
# Let's start with Time Series Forecasting using ARIMA model to forecast the number of subscriptions of Netflix. 

# In[17]:


#Convert the data into time series format

time_series = df.set_index('Time Period')['Subscribers']


# Here we are converting the original DataFrame into a time series format, where the Time Period column becomes the index, and the Subscribers column becomes the data.

# In[19]:


#Lets find the value of p and q by plotting
#the ACF and PACF of differenced time series:

differenced_series = time_series.diff().dropna()

#Plot ACF and PACF of differenced time series

fig, axes = plt.subplots(1,2, figsize =(12,4))
plot_acf(differenced_series, ax=axes[0])
plot_pacf(differenced_series, ax=axes[1])

plt.show()


# Here we first calculated the differenced time series from the original time_series, removed any NaN values resulting from the differencing, and then plotted the ACF and PACF to provide insights into the potential order of the AR and MA components in the time series. These plots are useful for determining the appropriate parameters when using the ARIMA model for time series forecasting.
# 
# Based on the plots, we find that p=1 and q=1. The ACF plot cuts off at lag 1, indicating q=1, and the PACF plot also cuts off at lag 1, indicating p=1. As there is a linear trend in the subscription growth rate, we can set the value of d as 1 to remove the linear trend, making the time series stationary.

# In[24]:


#Lets use the ARIMA model on our data:

p, d, q = 1, 1, 1
model = ARIMA(time_series, order =(p, d, q))
results = model.fit()
print(results.summary())


# In[25]:


#Lets make predictions using the trained model to forecast the number of 
#subscribers for the next five quarters:

future_steps = 6
predictions = results.predict(len(time_series), len(time_series)+ future_steps -1 )
predictions = predictions.astype(int)
print(predictions)


# Lets visualise the results of Netflix Subscriptions Forecasting for the next six quarters:

# In[29]:


#Creating a DataFrame with the original data and predictions
forecast = pd.DataFrame({'Original': time_series, 'Predictions': predictions})

#Plot the original data and predictions
fig = go.Figure()

fig.add_trace(go.Scatter(x=forecast.index, y =forecast['Predictions'],
                        mode ='lines', name = 'Predictions'))

fig.add_trace(go.Scatter(x=forecast.index, y=forecast['Original'],
                        mode = 'lines', name = 'Original Data'))

fig.update_layout(title = 'Netflix Quarterly Subscription Predictions',
                 xaxis_title = 'Time Period',
                 yaxis_title = 'Subscribers',
                 legend = dict(x=0.1, y=0.9),
                 showlegend = True)

fig.show()


# ## Summary
# 
# Using techniques like time series forecasting, Netflix can estimate the expected number of new subscribers in a given time period and better understand the growth potential of their business. It enhances operational efficiency, financial planning, and content strategy, ultimately contributing to their success and growth in the highly competitive streaming industry.
