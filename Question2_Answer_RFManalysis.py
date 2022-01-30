print("Hello, World!")
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

#load data into pandas dataframe..
dataset1 = pd.read_csv(r'D:\DATest_Handshakes\Log.csv')
dataset2 = pd.read_csv(r'D:\DATest_Handshakes\Users.csv')
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
frequency_log.columns = ['UserID', 'Frequency LogOn Times']
frequency_log.head()
print(frequency_log)

#Frequency based on online time interacting with the Intelligence portal
actlist = list(df['Action'])
timelist = list(df['CreatedAt'])
idlist = list(df['UserID'])
res = []
first, last = 0, 0
for i in range(len(actlist)):
    if i == 0:
        continue
    first = last + 1
    if actlist[i] == 'LogOn':
        last = i - 1
        onlinetime = (timelist[last] - timelist[first]).seconds
        res.append([idlist[i], onlinetime])
    if i == (len(actlist)-1):
        last = i
        onlinetime1 = (timelist[last] - timelist[first]).seconds
        res.append([idlist[i], onlinetime])
        
freq = pd.DataFrame(res, columns =['UserID', 'Online Time'])
print(freq)

frequency_onl = freq.groupby('UserID',as_index=False)['Online Time'].apply(lambda x: (sum(x))/3600)
print(frequency_onl)

frequency = frequency_log.merge(frequency_onl, on='UserID')
frequency.columns = ['UserID', 'Frequency LogOn', 'Frequency Online Time']
frequency.head()
print(frequency)

#Remove online times as it doesn't figure out the potential of users
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
rfm.head()
print(rfm)

#Set Recency and Frequency(LogOn and Online) quartiles:
def get_group1(Q1, Q2, Q3, value):
    if value < Q1:
        return '1'
    elif Q1 <= value < Q2:
        return '2'
    elif Q2 <= value < Q3:
        return '3'
    elif Q3 <= value:
        return '4'
fre_log_Q1, fre_log_Q2, fre_log_Q3 = rfm['Frequency LogOn'].quantile(
    [0.53, 0.666, 0.718])

def get_group2(Q1, Q2, Q3, value):
    if value > Q1:
        return '1'
    elif Q1 >= value > Q2:
        return '2'
    elif Q2 >= value > Q3:
        return '3'
    elif Q3 >= value:
        return '4'
rec_Q1, rec_Q2, rec_Q3 = rfm['Recency'].quantile(
    [0.25, 0.5, 0.75])
#fre_onl_Q1, fre_onl_Q2, fre_onl_Q3 = rfm['Frequency Online'].quantile([0.2743, 0.505, 0.693])

rec_group = []
for i in range(len(rfm)):
    rec = rfm.iloc[i]['Recency']
    rec_group.append(get_group2(rec_Q3, rec_Q2, rec_Q1, rec))
rfm['rec_group'] = rec_group
print(rec_Q3, rec_Q2, rec_Q1)

fre_log_group = []
for i in range(len(rfm)):
    fre_log = rfm.iloc[i]['Frequency LogOn']
    fre_log_group.append(get_group1(fre_log_Q1, fre_log_Q2, fre_log_Q3, fre_log))
rfm['fre_log_group'] = fre_log_group
print(fre_log_Q1, fre_log_Q2, fre_log_Q3)


#Compute rfm score and total score
rfm['RFM_score'] = rfm['rec_group'] + \
                   rfm['fre_log_group'] + \
                   rfm['Monetary'].astype(str)
rfm['Total_RFM_score'] = rfm['rec_group'].astype(int) + \
                         rfm['fre_log_group'].astype(int) + \
                         rfm['Monetary']
rfm.head()
print(rfm)

print(rfm.groupby('RFM_score')['Monetary'].mean())
print(rfm.groupby('Total_RFM_score')['Monetary'].mean())

#User segmentation based on Total RFM score
print("Best Users: ",len(rfm[rfm['Total_RFM_score']==12]))
print("Potential Users: ",len(rfm[rfm['Total_RFM_score'].between(9, 11)]))
print("Big Spenders: ",len(rfm[rfm['Monetary']==4]))
print("Frequent Users: ",len(rfm[rfm['fre_log_group']=='4']))
print("Small Spenders: ",len(rfm[rfm['Monetary']==1]))
print("Least Potential Users: ",len(rfm[rfm['RFM_score']=='111']))
print('Normal Users: ',len(rfm[rfm['Total_RFM_score'].between(4, 8)]))

rfm.query('Total_RFM_score > 8', inplace=True)
print(rfm)
