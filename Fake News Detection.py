#!/usr/bin/env python
# coding: utf-8

# # Fake News Detection

# ## Import Libraries

# In[33]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import re
import string


# ## Import Dataset

# In[3]:


df_fake = pd.read_csv("D:/Fake News Detection/Fake.csv")
df_true = pd.read_csv("D:/Fake News Detection/True.csv")
df_fake.head()


# In[4]:


df_true.head()


# ## Incoding the Binary Targets

# In[5]:


df_fake["class"] = 0
df_true["class"] = 1


# In[6]:


df_fake.shape, df_true.shape


# ## Merge Dataframes

# In[7]:


df_merge = pd.concat([df_fake, df_true], axis =0 )
df_merge.head()


# ## Remove Unwanted Cols

# In[8]:


df_merge.columns


# In[9]:


df = df_merge.drop(["title", "subject","date"], axis = 1)
df.head()


# ## Random Shuffling

# In[14]:


df = df.sample(frac = 1)
df.head()


# In[15]:


df.reset_index(inplace = True)
df.drop(["index"], axis = 1, inplace = True)
df.head()


# ## Check for Null

# In[16]:


df.isnull().sum()


# ## Cleaning and tokenization Function

# In[17]:


def preprocess(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text


# In[18]:


df["text"] = df["text"].apply(preprocess)
df.head()


# ## Define x for features and y for target

# In[19]:


x = df["text"]
y = df["class"]


# ## Train Test Split

# In[20]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)


# ## Features Extraction Using BOW-TFIDF

# In[23]:


tfidf = TfidfVectorizer()
x_train_tfidf = tfidf.fit_transform(x_train)
x_test_tfidf = tfidf.transform(x_test) 


# ## Logistic Regression

# In[25]:


LR = LogisticRegression()
LR.fit(x_train_tfidf,y_train)


# **Logistic Regression Evaluation**

# In[26]:


pred_lr=LR.predict(x_test_tfidf)
print(classification_report(y_test, pred_lr))


# ## Decision Tree

# In[28]:


DT = DecisionTreeClassifier()
DT.fit(x_train_tfidf, y_train)


# **Decision Tree Evaluation**

# In[29]:


pred_dt = DT.predict(x_test_tfidf)
print(classification_report(y_test, pred_dt))


# ## Ensemble Methods

# ### 1- Random Forest

# In[31]:


RF = RandomForestClassifier(random_state=0)
RF.fit(x_train_tfidf, y_train)


# **Evaluaiton**

# In[32]:


pred_rf = RF.predict(x_test_tfidf)
print(classification_report(y_test, pred_rf))


# ### 2- Gradient Boosting

# In[34]:


GB = GradientBoostingClassifier(random_state=0)
GB.fit(x_train_tfidf, y_train)


# **Evaluation**

# In[35]:


pred_gb = GB.predict(x_test_tfidf)
print(classification_report(y_test, pred_gb))


# 

# # The End!
# ## Best Regard,
# ### Ibrahim M. Nasser
