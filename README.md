# Online-Platform-user-segmentation
**I. Segment online users based on RFM analysis
**Firstly, to be more easy for analysis, we set dummies for Subscription Package in dataset:
• Package 1: 1
• Package 2: 2
• Package 3: 3
• Package 4: 4
1. Recency Determination
From CreatedAtcolumns we can calculate:
• Latest online time of each user
We set today is 15 July 2020.
We also assume that the greatest user recently logs on the Intelligence Portal and recency equals to 0. The better user just logged on the Portal one day before with recency is less than 1. The good user only logged on within a week and recency is less than 5. Other users latest logon time is more than a week and recency is greater than 5.
There fore, we have 3 quartiles of Recency: Q1 = 0, Q2 = 1, Q3 = 5
2. Frequency Determination
There are 2 indicators of frequency we can calculate:
• Average LogOntimes of each per day during dataset duration (13 days)
• Average Online time each user per day during dataset duration (13 days)
We assume that best users log on more than 2 times per day, better users log on from 1 to 2 times per day and good users log on from 0.5 to 1 time per day. The other users only log on less than 0.5 time per day on average. There fore, we have 3 quartiles of Frequency with LogOntimes: Q1 = 0.5, Q2 = 1, Q3 = 2
After consideration, we see that the average online time per day does not reflect how potential a user is so this indicator should be remove from the RFM analysis.
3. Monetary Determination
Base on the Subscription packages, we can calculate the monetary indicators:
• Package 1: 1
• Package 2: 2
• Package 3: 3
• Package 4: 4
4. RFM score
From Recency. Frequency and Monetary above we combine the calculated indicators to have the RFM analysis
To be more specific to evaluate our users, we create other tables:
• RFM score with average Subscription packages
• Total RFM score with average Subscription packages
From the Python code above, we can segment our users to 4 groups as followed:
• Best users: Users with Total RFM score equal to 12: 18 users
• Potential users: Users with Total RFM score from 9 to 11: 32 users
• Normal users: Users with Total RFM score from 4 to 8: 60 users
• Least potential users: Users with Total RFM score equal to 3: 5 users
Besides, we also have 3 other groups to pay attention on:
• Big spenders: Users with Monetary equal to 4: 58 users
• Frequent users: Users with Frequency equal to 4: 27 users
• Small spenders: Users with Monetary equal to 1: 21 users

**II. Data Inferential Analysis –Linear Regression
By speculation, the subscription package selection may depend on the Intelligence Portal usage level. For more detailed, user could select a higher subscription package based on the Logon times and frequency of use. Therefore, the variation of Monetary may depend on Recency and Frequency. Now, we will check the significance of this relationship:
We have an equation: Y = a.x1 + b.x2 + c
From Python, we will have the summary table of this linear regression model based on OLS method
From the summary:
• R-squared and adjusted R-squared approximately equal to 0: the variation of Recency and Frequency does not explain the variation of Monetary. This model does not fit
• P-value of Recency and Frequency respectively equal to 0.609 > 0.05 and 0.627> 0.05:these independent variables are not statistically significant.
