#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn import preprocessing 


# In[23]:


from sklearn.preprocessing import LabelEncoder


# In[2]:


df = pd.read_csv('/Users/eva-he/Downloads/Tableau Project - Supply Chain data/DataCoSupplyChainDataset.csv')


# In[3]:


df.head()


# In[ ]:


## Data Preprocessing 


# In[4]:


# combine first and last name
df['Customer Full Name'] = df['Customer Fname'].astype(str) + df['Customer Lname'].astype(str)
df= df.drop(['Customer Email','Product Status','Customer Password','Customer Street','Customer Fname','Customer Lname',
           'Latitude','Longitude','Product Description','Product Image','Order Zipcode','shipping date (DateOrders)'],axis=1)
df['Customer Zipcode'] = df['Customer Zipcode'].fillna(0)


# In[8]:


# Outliers
def outlier_treatment(col):
    sorted(col)
    Q1,Q3 = np.percentile(col,[25,75])
    IQR = Q3 - Q1
    lower_range = Q1 - (1.5*IQR)
    upper_range = Q3 + (1.5*IQR)
    return lower_range, upper_range
lower_range, upper_range = outlier_treatment(df['Product Price'])
df.drop(df[ (df['Product Price'] < lower_range) | (df['Product Price'] > upper_range)].index, inplace = True)

## .index to deal with dropping specific row 


# In[10]:


train_data = df.copy()
train_data['fraud'] = np.where(train_data['Order Status'] == 'SUSPECTED_FRAUD',1, 0)
train_data['late_delivery'] = np.where(train_data['Delivery Status'] == 'Late delivery', 1, 0)
train_data.drop(['Delivery Status','Late_delivery_risk','Order Status', 'order date (DateOrders)'], axis = 1, inplace = True)


# In[11]:


le = preprocessing.LabelEncoder()
train_data['Order Country'] = le.fit_transform(train_data['Order Country'])
train_data['Order State']    = le.fit_transform(train_data['Order State'])


# In[18]:


c_col = train_data.dtypes[(df.dtypes == 'object') | (train_data.dtypes == 'category')].index.tolist()


# In[19]:


c_col


# In[25]:


train_data[c_col] =  train_data[c_col].apply(LabelEncoder().fit_transform)


# In[ ]:


## Prediction Modeling 


# In[30]:


# Fraud Prediction
#Xf = train_data.loc[:, ~train_data.columns.isin(['fraud'])]
# Xf = train_data[['Days for shipping (real)','Days for shipment (scheduled)','Order Country','Category Name', 'Customer Segment']]
# yf = train_data['fraud']
# train_x, test_x, train_y, test_y = train_test_split(Xf,yf,test_size = 0.2, random_state = 42)
# random_forest = RandomForestClassifier(n_estimators = 100)
# random_forest.fit(train_x, train_y.values.ravel())
# random_forest.score(train_x, train_y)


# In[38]:


# Fraud Prediction
#Xf = train_data.loc[:, ~train_data.columns.isin(['fraud'])]
Xf=train_data[['Days for shipping (real)','Days for shipment (scheduled)','Order Country']]
yf = train_data['fraud']
train_x, test_x, train_y, test_y = train_test_split(Xf,yf,test_size = 0.2, random_state = 42)
random_forest = RandomForestClassifier(n_estimators = 100)
random_forest.fit(train_x, train_y.values.ravel())
random_forest.score(train_x, train_y)


# In[34]:


# Late Delivery Prediction
#Xl = train_data.loc[:, ~train_data.columns.isin(['fraud'])]
# Xl = train_data[['Days for shipment (scheduled)','Order Country','Category Name']]
# yl = train_data['late_delivery']
# train_xl, test_xl, train_yl, test_yl = train_test_split(Xl,yl,test_size = 0.2, random_state = 42)
# random_forest_1 = RandomForestClassifier(n_estimators = 100)
# random_forest_1.fit(train_xl, train_yl.values.ravel())
# random_forest_1.score(train_xl, train_yl)


# In[40]:


Xl=train_data[['Days for shipment (scheduled)','Order Country']]
yl=train_data['late_delivery']
train_xl,test_xl,train_yl,test_yl = train_test_split(Xl,yl,test_size = 0.2, random_state = 42)
random_forest_l = RandomForestClassifier(n_estimators=100)
random_forest_l.fit(train_xl, train_yl.values.ravel())
random_forest_l.score(train_xl, train_yl)


# In[32]:


import tabpy_client
from tabpy.tabpy_tools.client import Client
client = tabpy_client.Client('http://localhost:9004/')


# In[33]:


# def fraud_predictor5(_arg1,_arg2,_arg3,_arg4,_arg5):
#     import pandas as pd 
#     row = {'shipping': _arg1,
#           'shipping scheduled': _arg2,
#           'country': _arg3,
#           'Category': _arg4,
#           'Customer Segment': _arg5}
#     test_data = pd.DataFrame(data = row, index = [0])
#     from sklearn import preprocessing 
#     le = preprocessing.LabelEncoder()
#     c_cols = ['country','Category','Customer Segment']
#     test_data[['country','Category','Customer Segment']] =  test_data[c_cols].apply(le.fit_transform)
#     # Predict the fraud 
#     predprob = random_forest.predict_proba(test_data)
#     # Return only the probability 
#     return[probability[1] for probability in predprob]


# In[35]:


# def late_delivery(_arg1, _arg2, _arg3):
#     import pandas as pd
#     row = {'shipping scheduled': _arg1,
#           'country': _arg2,
#           'Category Name': _arg3}
#     test_data = pd.DataFrame(data = row, index = [0])
#     from sklearn import preprocessing
#     le = preprocessing.LabelEncoder()
#     test_data[['country','Category Name']] = test_data[['country','Category Name']].apply(le.fit_transform)
#     predprob = random_forest_1.predict_proba(test_data)
#     return[probability[1] for probability in predprob]


# In[41]:


def fraud_predictor5( _arg1, _arg2,_arg3):
    import pandas as pd
    row = {'shipping': _arg1,
           'shipping scheduled': _arg2,
          'country_str':_arg3}
    #Convert it into a dataframe
    test_data = pd.DataFrame(data = row,index=[0])
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    test_data['country_str']  = le.fit_transform(test_data['country_str'])
    #Predict the Fraud
    predprob_survival = random_forest.predict_proba(test_data)
    #Return only the probability
    return [probability[1] for probability in predprob_survival]


# In[42]:


def late_delivery( _arg1, _arg2):
    import pandas as pd
    row = {'shipping scheduled': _arg1,
          'country_str':_arg2}
    #Convert it into a dataframe
    test_data = pd.DataFrame(data = row,index=[0])
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    test_data['country_str']  = le.fit_transform(test_data['country_str'])
    #Predict the late delivery probabilites
    predprob_late = random_forest_l.predict_proba(test_data)
    #Return only the probability
    return [probability[1] for probability in predprob_late]


# In[43]:


# Deploying
client.deploy('fraud_predictor5', fraud_predictor5,'fraud_predictor probability', override = True)


# In[44]:


client.deploy('late_delivery', late_delivery,'late_delivery_probability',override = True)


# In[ ]:




