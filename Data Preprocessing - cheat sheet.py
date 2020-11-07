#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('wine.csv', sep=';')


# In[3]:


df.head(2)


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.shape


# In[7]:


# Scatter Matrix, remember to add the ";" at the end and to not make it very noisy

pd.plotting.scatter_matrix(df,alpha = 0.2, figsize = [15,10]);


# In[8]:


# one nice way to print the head by transposing it. Now each column become a row

first_rows = df.head(3).transpose()
print(first_rows)


# In[9]:


# Get descriptions of every column, again transposed for different type of visualizng it

col_descriptions = df.describe(include = 'all',
                                    percentiles = [0.5]).transpose()
print(col_descriptions)


# ### Removing missing Data

# In[10]:


df.dropna()


# In[11]:


## droping specific rows
# df.drop([1,2])

## drop specific column
# df.drop('pH', axis = 1)

## checking if specific column has 0 value
df['pH'].isnull().sum()


# In[12]:


## filtering on rows that has null in specific column (or could be for all if we want)
df[df['pH'].notnull()]

## we can either just print it out or put it in another dataFrame
df_not_null = df[df['pH'].notnull()]
df_not_null.head(3)


# ### Working with Data type

# In[13]:


print(df.dtypes)


# In[14]:


## Change the data type of specific column is straightforward

df['citric acid'] = df['citric acid'].astype("object")
print(df['citric acid'].dtypes)


# In[15]:


df['citric acid'] = df['citric acid'].astype("float64")
print(df['citric acid'].dtypes)


# In[20]:


## get columns names
df.columns


# ## Splitting dataset into training and testing

# In[21]:


from sklearn.model_selection import train_test_split


# In[22]:


train, test = train_test_split(df, test_size = 0.2, random_state = 4)
## add a random_state point (4) so results can be reproduced
## After spliting the data into train and test, then we split them further into x_train,x_test, y_train, y_test


# In[23]:


## A Faster way is to get the column names split based on that

y = train["quality"]
cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']
X=train[cols]


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 4)


# Faster way to get X, Y from your dataframe

# In[27]:


# Create a data with all columns except quality which is our target Y
df_X = df.drop("quality", axis=1)

# Create a quality labels dataset
df_y = df[["quality"]]


# #### Splitting more unbalanced Dataset, using stratified sampling

# In[30]:


X_train, X_test, y_train, y_test = train_test_split(X,y, stratify = y)


# In[31]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# ### Standarizing Dataset before processing it in to ML model

# #### Standarization is preprocessing step in which we transform continous nomerical data to make look normally distributed. it is important steps since most ML models assume our data are normally distributed

# ##### 1- LogNormalization: can be useful when having particular column with high variance. good if you still want to capture the magnitude of change and Keep everything in POSITIVE SPACE
# ##### 2- Feature scalling: it tranform the data/column to have mean of 0 and variance of 1 (it will make it easier to lineary compare features). One good usage is when having dataset where datas within the same column are relatively close in scale but they differe across different columns

# In[32]:


df.var()


# In[33]:


## has significatly high variance free sulfur dioxide, total sulfur dioxide

df["Log_free sulfur dioxide"] = np.log(df["free sulfur dioxide"]) 
df["Log_total sulfur dioxide"] = np.log(df["total sulfur dioxide"]) 


# In[34]:


df.var()


# In[35]:


## Using Scalling

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[36]:


df_scaled = pd.DataFrame(scaler.fit_transform(df), columns = df.columns)


# In[37]:


df_scaled.head(3)


# In[38]:


print(df_scaled.var())
print(df_scaled.mean())


# ### Feature engineering

# #### Encoding

# In[39]:


## let's create a dataFrame on the fly that we can do some Ebcoding on
data = {'user':  [1,2,3,4],
        'subscribed': ['y', 'n','n','y'],
        'fav_color': ['blue', 'green','red','white'],
        }

new_df = pd.DataFrame (data, columns = ['user','subscribed','fav_color'])
new_df.head()


# In[40]:


## The long way using Pandas

new_df['subscribed_encoded'] = new_df['subscribed'].apply(lambda val: 1 if val=='y' else 0)


# In[41]:


print(new_df[['subscribed','subscribed_encoded']])


# In[42]:


## Using Sklearn

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
new_df['sub_sklearn_enc'] = le.fit_transform(new_df['subscribed'])


# In[43]:


print(new_df[['subscribed','sub_sklearn_enc']])


# In[44]:


## OneHotEncoding is also good when you have more than one value to encode. Like here in fav_color column we have 4 different colors

print(pd.get_dummies(new_df['fav_color']))


# In[46]:


## One practical way is to use get_dummies to create encoding for the categorical column, store it in variable and then merge it back into the DF

# Create a one-hot encoded set of the type values
fav_colors = pd.get_dummies(new_df["fav_color"])

# Concatenate this set back to the ufo DataFrame
ufo = pd.concat([new_df, fav_colors], axis=1)
ufo.head()


# #### Feature engineering to nomerical features

# In[47]:


## replacing a several columns with average instead. Lets say intead of "fixed acidity" "volatile acidity" i would like to take mean instead

col = ["fixed acidity","volatile acidity"]
df["mean"] = df.apply(lambda row: row[col].mean(), axis = 1)
df.head()


# #### Working with date

# In[48]:


## Creating new DF that have date so we can play with it

data = {'date':  ['February 05 2011','February 14 2011','January 29 2011','February 01 2011'],
        'subscribed': ['y', 'n','n','y'],
          }

new_df = pd.DataFrame(data, columns = ['date','subscribed'])  


# In[49]:


## I don't have date column here but technically i will first make sure the column is date type. then extract what i want, like month, year...etc

new_df["date_new_if_needed"] = pd.to_datetime(new_df['date'])
new_df["month"] = new_df.date_new_if_needed.apply(lambda row: row.month)


# ### Regular expression

# In[50]:


import re


# In[51]:


## Creating new DF that have date to manipulate

data = {'Length':  ['0.5 miles','0.5 miles','0.75 miles','1.0 mile'],
        'makeNoSenseData': ['Fight global hunger and support women farmers', 'Urban Adventures - Ice Skating at Lasker Rink',' Web designer','Volunteers Needed For Rise Up & Stay Put! Home'],
          }

new_df = pd.DataFrame(data, columns = ['Length','makeNoSenseData']) 

new_df


# In[53]:


## to extract the number 0.5 (float) we use the below compile
# Write a pattern to extract numbers and decimals
def return_mileage(length):
    pattern = re.compile(r"\d+\.\d+")
    
    # Search the text for matches
    mile = re.match(pattern, length)
    
    # If a value is returned, use group(0) to return the found value
    if mile is not None:
        return float(mile.group(0))
        
# Apply the function to the Length column and take a look at both columns
new_df["Length_num"] = new_df["Length"].apply(lambda row: return_mileage(row))
print(new_df[["Length", "Length_num"]].head())


# ### Feature selection
# 
# #### it is good to take a look at Sklearn statistical methods for feature selection. here we'll look at how to manually select features. If features are highly correlated then some of them will need to be removed otherwise we'll introduce bias into our model

# In[54]:


col_ = ["fixed acidity","volatile acidity","mean"]
df_new = df[col_]


# In[55]:


print(df_new.corr())


# In[56]:


## since mean and fixed acidity are highly correlated, we can drop one of them
to_drop = "mean"
df_new = df_new.drop(to_drop, axis=1)
df_new.head()


# ### Dimensionality reduction

# #### PCA: use linear tranformation into uncorrelated space. while the features are reduced however still capture as much variance as possible in each component (combining features into components) - it is good way when we have large number of features and we donät have strong candidates for eleminations. It is good to know that it is kinda black box in comparison to other Dimensionality reduction methods, and good to do it at the end of preprocessing steps

# In[57]:


from sklearn.decomposition import PCA

pca = PCA()
df_pca = pca.fit_transform(df)

print(df_pca)


# In[58]:


print(pca.explained_variance_ratio_)


# The bigger the number the higher the chunck of variance explanation. 
# higher explained_variance_ratio_ means much of the variance are explained by this component
# So likely to drop those component that don't explain much variance

# In[59]:


print("biggest one: ", max(pca.explained_variance_ratio_))
print("All of them sorted from highest to lowest: ", sorted(pca.explained_variance_ratio_, reverse = True))

