print("Hello, World!")
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

#load data into pandas dataframe..
dataset1 = pd.read_csv(r'D:\Log.csv')
dataset2 = pd.read_csv(r'D:\Users.csv')
dataset = dataset1.merge(dataset2, on='UserID', how='inner')
dataset = dataset.drop_duplicates()
df = pd.DataFrame(dataset)
df.head()

#information of dataset
df.info()

#Recency estimation
df['CreatedAt'] = pd.to_datetime(df['CreatedAt'])
print(df['CreatedAt'])

max = df['CreatedAt'].max()
min = df['CreatedAt'].min()
duration = (df['CreatedAt'].max() - df['CreatedAt'].min()).days
print(duration)

now = dt.datetime(2020, 7, 15, 00, 00, 00)
print(now)
                       
recency = df.groupby(['UserID'],as_index=False)['CreatedAt'].max()
recency.columns = ['UserID', 'LastOnlineTime']
recency.head()
print(recency)

#Calculate how often a user is online with reference to latest date & time in days..
recency['Recency'] = recency['LastOnlineTime'].apply(
    lambda x: (now - x).days)
recency.head()
print(recency)

#Frequency estimation: How often a user is online daily by logOn times and online time
#Frequency based on Log on times:
frequency_log = df.groupby(['UserID'],as_index=False)['Action'].apply(
    lambda x: (
        (sum(1 for x in list(x) if x=='Token'))/13))
frequency_log.columns = ['UserID', 'Frequency']
frequency_log.head()
print(frequency_log)

#Monetary estimation: Package
monetary = df.groupby(['UserID'],as_index=False)['Packages'].max()
monetary.columns = ['UserID', 'Monetary']
monetary.head()
print(monetary)

#RFM estimation based on Recency, Frequency and Monetary
rf = recency.merge(frequency_log, on='UserID').drop(columns='LastOnlineTime')
rfm = rf.merge(monetary, on='UserID')
rfm.columns = ['UserID', 'Recency',
               'Frequency LogOn', 'Monetary']

#Linear Regression model with 2 independent variable: Freqency & Recency:
#define data
x = rfm[['Frequency LogOn','Recency']]
y = rfm['Monetary']

#linear regression model
model = LinearRegression()
model.fit(x, y)
model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)

#print summary
X = sm.add_constant(x)
est = sm.OLS(y, X)
est2 = est.fit()
print(est2.summary())
