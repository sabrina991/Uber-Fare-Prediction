#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis and Visualization 

# In[4]:


# import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pylab
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn import metrics
import math
import plotly.express as px
from statsmodels.tools.eval_measures import rmse

from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV


# Read data

# In[5]:


# Here we are importing dataset by using pandas
data = pd.read_csv('uber.csv')
data


# In[6]:


# always create a copy 
df = data.copy()


# In[7]:


# see the complete dataframe
pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)

df.head()


# In[8]:


df.info()


# In[9]:


df.columns


# In[10]:


df = df.drop(['Unnamed: 0', 'key'], axis= 1)

df.shape


# In[11]:


df.dtypes


# In[12]:


df.info()


# In[13]:


df.shape


# In[16]:


df.describe()


# Visualization

# In[109]:


plt.scatter(df['passenger_count'], df['fare_amount'])
plt.xlabel('Passenger Count')
plt.ylabel('Fare Amount')
plt.title('Scatter Plot: Passenger Count vs. Fare Amount')
plt.show()


# In[110]:


plt.hist(df['fare_amount'], bins=20)
plt.xlabel('Fare Amount')
plt.ylabel('Frequency')
plt.title('Histogram: Fare Amount Distribution')
plt.show()


# In[111]:


df.isnull().sum()


# In[18]:


df.head()


# In[19]:


df.dtypes


# change the Incorrect data type

# In[20]:


df.pickup_datetime = pd.to_datetime(df.pickup_datetime, errors='coerce') 
# # For datetime64[ns] types, NaT represents missing values.


# Recheck the data type

# In[21]:


df.dtypes


# In[22]:


# we will extract time feature from the 'pickup_datetime' 
# we will add a variable which measures the distance between pickup and drop

df= df.assign(hour = df.pickup_datetime.dt.hour,
             day= df.pickup_datetime.dt.day,
             month = df.pickup_datetime.dt.month,
             year = df.pickup_datetime.dt.year,
             dayofweek = df.pickup_datetime.dt.dayofweek)


# Here we are going to use Heversine formula to calculate the distance between two points and journey, using the longitude and latitude values.

# In[23]:


df.info()


# In[24]:


from math import *


# Heversine formula¶
# hav(θ) = sin**2(θ/2).

# In[25]:


# function to calculate the travel distance from the longitudes and latitudes
def distance_transform(longitude1, latitude1, longitude2, latitude2):
    travel_dist = []
    
    for pos in range(len(longitude1)):
        long1,lati1,long2,lati2 = map(radians,[longitude1[pos],latitude1[pos],longitude2[pos],latitude2[pos]])
        dist_long = long2 - long1
        dist_lati = lati2 - lati1
        a = sin(dist_lati/2)**2 + cos(lati1) * cos(lati2) * sin(dist_long/2)**2
        c = 2 * asin(sqrt(a))*6371
        travel_dist.append(c)
       
    return travel_dist

df['dist_travel_km'] = distance_transform(df['pickup_longitude'].to_numpy(),
                                                df['pickup_latitude'].to_numpy(),
                                                df['dropoff_longitude'].to_numpy(),
                                                df['dropoff_latitude'].to_numpy()
                                              )

#check the dataset
df.head()


# In[26]:


# now we can drop the pickup_datetime variable beacause now we got the distance 
#and it becomes redundant now.
# drop the column 'pickup_daetime' using drop()
# 'axis = 1' drops the specified column

df = df.drop('pickup_datetime',axis = 1)


# In[27]:


df.info()


# In[28]:


# as now we get the required variable and in the required in datatype so now we can move further.

df.describe().transpose()


# In[29]:


df.columns[df.dtypes == 'object']


# # Preprocessing, Feature Engineering 

# Outlier

# In[30]:


df.head()


# In[31]:


df.fare_amount.min()


# In[32]:


#sns.set(rc={'figure.figsize':(11.7,8.27)})

plt.figure(figsize=(20,12))
sns.boxplot(data= df)


# In[33]:


medianFiller = lambda x : x.fillna(x.median())
numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
df[numeric_columns] = df[numeric_columns].apply(medianFiller, axis = 0)

# outliers detection using boxplot
plt.figure(figsize =(20, 30))

for i , variable in enumerate(numeric_columns):
    plt.subplot(6 , 5, i +1)
    plt.boxplot(df[variable], whis=1.5)
    plt.tight_layout()
    plt.title(variable)
    
plt.show()


# In[34]:


df.shape


# In[35]:


#We will only keep the observation where fare is between 2.5
df = df.loc[(df.fare_amount >= 0)]

df.shape


# In[36]:


df.head()


# In[37]:


def remove_outlier(df1 , col):
    Q1 = df1[col].quantile(0.25)
    Q3 = df1[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_whisker = Q1-1.5*IQR
    upper_whisker = Q3+1.5*IQR
    df[col] = np.clip(df1[col] , lower_whisker , upper_whisker)
    return df1

def treat_outliers_all(df1 , col_list):
    for c in col_list:
        df1 = remove_outlier(df , c)
    return df1

df = treat_outliers_all(df , df.iloc[: , 0::])

plt.figure(figsize=(20,30))
for i , variable in enumerate(df.iloc[: , 0::]):
    plt.subplot(6,5,i+1)
    plt.boxplot(df[variable] , whis = 1.5)
    plt.tight_layout()
    plt.title(variable)
plt.show()


# In[38]:


df.shape


# In[39]:


#Remove the observations with travel distance more than 130 kms.As seen from descriptive stats there are obs with travel 
#distance more than 130 km, as that is the limit for trips.

#We will only keep the observation where travel distance is less than or equal to 130
df= df.loc[(df.dist_travel_km >= 1) | (df.dist_travel_km <= 130)]
print("Remaining observastions in the dataset:", df.shape)


# In[40]:


#Remove the observations with unreal longitude and latitude values , that is , 90< latitude <-90, 180 < longitude < -180

incorrect_coordinates = df.loc[(df.pickup_latitude > 90) |(df.pickup_latitude < -90) |
                                   (df.dropoff_latitude > 90) |(df.dropoff_latitude < -90) |
                                   (df.pickup_longitude > 180) |(df.pickup_longitude < -180) |
                                   (df.dropoff_longitude > 90) |(df.dropoff_longitude < -90)
                                    ]

df.drop(incorrect_coordinates, inplace = True, errors = 'ignore')

df.head()


# In[41]:


df.shape


# Missing Values

# In[42]:


# sort the variables on the basis of total null values in the variable
# 'isnull().sum()' returns the number of missing values in each variable
# 'ascending = False' sorts values in the descending order
# the variable with highest number of missing values will appear first
Total = df.isnull().sum().sort_values(ascending = False)          

# calculate the percentage of missing values
# 'ascending = False' sorts values in the descending order
# the variable with highest percentage of missing values will appear first
Percent = (df.isnull().sum()*100/df.isnull().count()).sort_values(ascending = False)   

# concat the 'Total' and 'Percent' columns using 'concat' function
# 'keys' is the list of column names
# 'axis = 1' concats along the columns
missing_data = pd.concat([Total, Percent], axis = 1, keys = ['Total', 'Percentage of Missing Values']) 

# add the column containing data type of each variable
missing_data['Type'] = df[missing_data.index].dtypes
missing_data


# In[43]:


#visulize the missing values by heatmap

# plot heatmap to visualize the null values in each column
# 'cbar = False' does not show the color axis 
sns.heatmap(df.isnull())

# display the plot
plt.show()


# In[44]:


#compute a correlation matrix

# use the corr() function to generate the correlation matrix of the numeric variables
corr = df.corr()

# print the correlation matrix
corr


# In[45]:


#Plot the heatmap for the diagonal correlation matrix

# set the plot size
# pass the required height and width to the parameter, 'figsize'  
plt.figure(figsize = (30,20))

# use 'mask' to plot a upper triangular correlation matrix 
# 'tril_indices_from' returns the indices for the lower-triangle of matrix
# 'k = -1' consider the diagonal of the matrix
mask = np.zeros_like(corr)
mask[np.tril_indices_from(mask, k = -1)] = True

sns.heatmap(corr, cmap = 'RdYlGn', vmax = 1.0, vmin = -1.0, annot = True, annot_kws = {"size": 20}, mask = mask)

# set the size of x and y axes labels
# set text size using 'fontsize'
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 15)

# display the plot
plt.show()


# In[46]:


# set figure size
plt.figure(figsize=(30, 20))

sns.heatmap(corr[(corr >= 0.7) | (corr <= -0.7)], cmap = 'RdYlGn', vmax = 1.0, vmin = -1.0, annot = True, 
            annot_kws={"size": 20})

# set the size of x and y axes labels
# set text size using 'fontsize'
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)

# display the plot
plt.show()


# 1. Plot a countplot to check the peak hours during weekdays

# In[47]:


# Set the figure size, grid type and color palette along with appropraite titel for the plot
plt.figure(figsize=(10,5))
plt.title('Cab trips count during week days')
sns.set_style("white")
#plot the countplot for weekdays
sns.countplot(x='hour', data=df.loc[(df.dayofweek >= 0) & (df.dayofweek <=4)], palette='Set2')


# 2. Plot a countplot to check the peak hours during weekends

# In[48]:


# Set the figure size, grid type and color palette along with appropraite titel for the plot
plt.figure(figsize=(10,5))
plt.title('Cab trips count during week ends')
sns.set_style("white")
#plot the countplot for weekends
sns.countplot(x='hour', data=df.loc[(df.dayofweek >= 5) & (df.dayofweek <=6)], palette='Set2')


# Compute the hourly mean fare for weekdays and weekends

# In[49]:


# Set the days in the dataset as week days and week ends
week_days = df.loc[(df.dayofweek >= 0) & (df.dayofweek <= 4)]
week_ends = df.loc[(df.dayofweek >= 5) & (df.dayofweek <= 6)]

# compute the mean fare amount over the week day and week end.
# use groupby('hour') to get the mean fare for each hour
week_days_fare = week_days.groupby(['hour']).fare_amount.mean().to_frame().reset_index()
week_ends_fare = week_ends.groupby(['hour']).fare_amount.mean().to_frame().reset_index()


# Plot the mean fare

# In[50]:


# hours
x = np.array(week_days_fare.hour)
# an array of week day fare 
y = np.array(week_days_fare.fare_amount)
# an array of week end fare
z = np.array(week_ends_fare.fare_amount)

# Set the figure size, title, x and y labels
plt.figure(figsize = (20,10))
plt.title('Mean Fare Amount For Each Hour - Weekdays Vs Weekends')
plt.xlabel('Hours')
plt.ylabel('Mean Fare')
# Pass the three integers. The value of these integer should be less that 10
ax=plt.subplot(1,1,1)
ax.bar(x-0.2, y, width=0.2, color='red', align='center', label = 'Week days')
ax.bar(x, z, width=0.2, color='blue', align='center', label = 'Week ends')
plt.xticks(range(0,24))
plt.legend()
plt.show()


# Distribution of "fare_amount"

# In[51]:


plt.figure(figsize=(20,10))
sns.set_style("darkgrid")
plt.title("Distribution of the fare amount")
plt.xlabel("Fare Amount")
plt.ylabel("Frequency")
plt.xlim(-10,20)
plt.xticks(range(0,200,5))

snsplot = sns.kdeplot(df.fare_amount, shade=True)


# Distribustion of the travel distance in km

# In[52]:


plt.figure(figsize = (20,10))
sns.style = ('darkgrid')
plt.title("Distribution of the trip distance")
plt.xlabel("Distance(Km)")
plt.ylabel("Frequency")
plt.xlim(-10, 200)
plt.xticks(range(0,200,5))

sns.plot = sns.kdeplot(df[df.dist_travel_km<600].dist_travel_km, shade=True)


# # Feature Extraction & Model Building and Performance Measure 

# 1. Split the data into features and target dataframes

# In[53]:


# select only the target variable 'amount' and store it in dataframe 'y'
y = pd.DataFrame(df['fare_amount'])

# use 'drop()' to remove the variable 'amount' from df_taxi
# 'axis = 1' drops the corresponding column(s)
x = df.drop('fare_amount',axis = 1)

x.head()


# 2. Split the data into training and test sets

# In[54]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

# check the dimensions of the train & test subset for 
# print dimension of predictors train set
print("The shape of X_train is:",x_train.shape)

# print dimension of predictors test set
print("The shape of X_test is:",x_test.shape)

# print dimension of target train set
print("The shape of y_train is:",y_train.shape)

# print dimension of target test set
print("The shape of y_test is:",y_test.shape)


# 1.OLS

# In[56]:


linreg_full = sm.OLS(y_train, x_train).fit()

# print the summary output
print(linreg_full.summary())


# In[60]:


linreg_full_predictions = linreg_full.predict(x_test)
linreg_full_predictions


# In[61]:


cols = ['Model', 'RMSE', 'R-Squared', 'Adj. R-Squared']

# create an empty dataframe with the specified columns
result_tabulation = pd.DataFrame(columns=cols)

# Define values for the variables
linreg_full_rmse = 0.123  # Replace with the actual RMSE value
linreg_full_rsquared = 0.456  # Replace with the actual R-Squared value
linreg_full_rsquared_adj = 0.789  # Replace with the actual Adjusted R-Squared value

# Compile the required information
linreg_full_metrics = pd.Series({'Model': "Linreg full model",
                                 'RMSE': linreg_full_rmse,
                                 'R-Squared': linreg_full_rsquared,
                                 'Adj. R-Squared': linreg_full_rsquared_adj
                                })

# Append the metrics to the result_tabulation DataFrame
result_tabulation = result_tabulation.append(linreg_full_metrics, ignore_index=True)

# Print the result table
print(result_tabulation)


# 2.Linear Regression with SGD

# In[62]:


# Importing necessary libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Scale the data using StandardScaler
scaler = StandardScaler()
scaler.fit(x)


# In[63]:


x1= scaler.fit_transform(x)
x1


# In[64]:


y1=y
y1


# In[65]:


x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size = 0.2, random_state = 1)

# check the dimensions of the train & test subset for 
# print dimension of predictors train set
print("The shape of X1_train is:",x1_train.shape)

# print dimension of predictors test set
print("The shape of X1_test is:",x1_test.shape)

# print dimension of target train set
print("The shape of y1_train is:",y1_train.shape)

# print dimension of target test set
print("The shape of y1_test is:",y1_test.shape)


# In[66]:


SGD_model = SGDRegressor(loss="squared_error", alpha=0.1, max_iter=1000)  # Instantiate the model

# Fit the model
SGD_model.fit(x1_train, y1_train)


# In[67]:


# predict the values
y1_pred_SGD  = SGD_model.predict(x1_test)
y1_pred_SGD


# In[68]:


r_squared_SGD = SGD_model.score(x1_train,y1_train)
# Number of observation or sample size
n = 159999 

# No of independent variables
p = 11

#Compute Adj-R-Squared
Adj_r_squared_SGD = 1 - (1-r_squared_SGD)*(n-1)/(n-p-1)

# Compute RMSE
rmse_SGD = sqrt(mean_squared_error(y1_test, y1_pred_SGD))


# In[69]:


linreg_full_metrics = pd.Series({'Model': "Linear regression with SGD",
                     'RMSE':rmse_SGD,
                     'R-Squared': r_squared_SGD,
                     'Adj. R-Squared': Adj_r_squared_SGD     
                   })

# append our result table using append()
# ignore_index=True: does not use the index labels
# python can only append a Series if ignore_index=True or if the Series has a name
result_tabulation = result_tabulation.append(linreg_full_metrics, ignore_index = True)

# print the result table
result_tabulation


# 3. Decision Tree

# In[70]:


from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV


# In[71]:


#decision_tree = DecisionTreeRegressor(criterion = 'mse', random_state = 10) #Max depth D.Tree gets formed
decision_tree = DecisionTreeRegressor(criterion='squared_error', random_state=10) #Max depth D.Tree gets formed

# fit the model using fit() on train data
decision_tree_model = decision_tree.fit(x_train, y_train)


# In[72]:


y_pred_DT=decision_tree_model.predict(x_test)

y_pred_DT


# In[73]:


y_test


# In[74]:


r_squared_DT=decision_tree_model.score(x_test,y_test)
# Number of observation or sample size
n = 159999 

# No of independent variables
p = 11

#Compute Adj-R-Squared
Adj_r_squared_DT = 1 - (1-r_squared_DT)*(n-1)/(n-p-1)
Adj_r_squared_DT


# In[75]:


# Compute RMSE
rmse_DT = sqrt(mean_squared_error(y_test, y_pred_DT))


# In[76]:


# compile the required information
linreg_full_metrics = pd.Series({'Model': "Decision Tree Model ",
                     'RMSE':rmse_DT,
                     'R-Squared': r_squared_DT,
                     'Adj. R-Squared': Adj_r_squared_DT     
                   })

# append our result table using append()
# ignore_index=True: does not use the index labels
# python can only append a Series if ignore_index=True or if the Series has a name
result_tabulation = result_tabulation.append(linreg_full_metrics, ignore_index = True)

# print the result table
result_tabulation


# 4.Decision Tree with pruning

# In[77]:


prune = DecisionTreeRegressor(max_depth = 10, max_leaf_nodes = 32 , random_state = 10)

# fit the model using fit() on train data
decision_tree_prune = prune.fit(x_train, y_train)

y_pred_DT_prune=decision_tree_prune.predict(x_test)

y_pred_DT_prune


# In[78]:


r_squared_DT_prune=decision_tree_prune.score(x_test,y_test)
# Number of observation or sample size
n = 159999  

# No of independent variables
p = 11

#Compute Adj-R-Squared
Adj_r_squared_DT_prune = 1 - (1-r_squared_DT_prune)*(n-1)/(n-p-1)
Adj_r_squared_DT_prune
# Compute RMSE
rmse_DT_prune = sqrt(mean_squared_error(y_test, y_pred_DT_prune))

# compile the required information
linreg_full_metrics = pd.Series({'Model': "Decision Tree Model after pruning ",
                     'RMSE':rmse_DT_prune,
                     'R-Squared': r_squared_DT_prune,
                     'Adj. R-Squared': Adj_r_squared_DT_prune     
                   })

# append our result table using append()
# ignore_index=True: does not use the index labels
# python can only append a Series if ignore_index=True or if the Series has a name
result_tabulation = result_tabulation.append(linreg_full_metrics, ignore_index = True)

# print the result table
result_tabulation


# Finding the MSE,MAE, RMSE, etc

# In[71]:


from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
#print('Mean Absolute % Error:', metrics.mean_absolute_percentage_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[115]:


from sklearn.metrics import mean_squared_error, r2_score

# Split the data into training and test sets (assuming you haven't already done this).
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train a machine learning model (e.g., linear regression).
model = LinearRegression()
model.fit(x_train, y_train)

# Make predictions on the test set.
y_pred = model.predict(x_test)

# Calculate the R2 score and mean squared error.
accuracy = r2_score(y_test, y_pred)
error = mean_squared_error(y_test, y_pred)

print("Accuracy:", accuracy)
print("Mean Squared Error:", error)


# # Model Interpretability

# Using LIME

# In[98]:


pip install lime


# In[100]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Assuming X contains your feature data and y contains the fare amounts.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)


# In[101]:


from lime.lime_tabular import LimeTabularExplainer


# In[105]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train a machine learning model (e.g., linear regression).
model = LinearRegression()
model.fit(x_train, y_train)

# Create a LIME explainer.
explainer = LimeTabularExplainer(x_train.values, mode="regression", training_labels=y_train, feature_names=x_train.columns)

# Select a prediction to explain (e.g., the first instance in the test set).
instance_to_explain = x_test.iloc[0]

# Generate explanations.
explanation = explainer.explain_instance(instance_to_explain, model.predict, num_features=5)

# Visualize the explanation.
explanation.show_in_notebook()


# # Model Deployment

# In[119]:


get_ipython().system('pip install streamlit')


# In[124]:


st.write("""
# Uber Fare Visualization
Explore your Uber fare data.
""")

# Allow the user to select columns for the x and y axes
x_column = st.selectbox("Select x-axis data", df.columns)
y_column = st.selectbox("Select y-axis data", df.columns)

if x_column != y_column:
    st.line_chart(df[[x_column, y_column]])
else:
    st.write("Please select different columns for x and y axes.")


# In[116]:


import pickle


# In[117]:


filename = 'trained_model.sav'
pickle.dump(scaler, open(filename, 'wb'))


# In[118]:


#loading the saved model

loaded_model = pickle.load(open('trained_model.sav', 'rb'))


# In[ ]:




