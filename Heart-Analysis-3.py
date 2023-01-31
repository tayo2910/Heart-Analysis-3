#!/usr/bin/env python
# coding: utf-8
import streamlit as st
st.title('Heart Disease Analysis')

st.markdown('Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worldwide. Four out of 5CVD deaths are due to heart attacks and strokes, and one-third of these deaths occur prematurely in people under 70 years of age. Heart failure is a common event caused by CVDs and this dataset contains 11 features that can be used to predict a possible heart disease, Let/s dive in')


# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# In[2]:

st.write('Reading the Data')
data = pd.read_csv('heart.csv')
data


# In[3]:

st.write('Checking Duplicates')
c = data.duplicated().sum()
c
st.write('Checking for null values')
d = data.isna().sum()
d


# ## Transforming the Categorical columns

# In[6]:


data['ChestPainType'].unique()


# In[7]:


data['Sex'].unique()


# In[8]:


data['RestingECG'].unique()


# In[9]:


data['ExerciseAngina'].unique()


# In[10]:


data['ST_Slope'].unique()


# In[11]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# In[12]:


encoder = LabelEncoder()


# In[13]:
st.write('Changing Nominal to Ordinal values')

data[['ChestPainType', 'RestingECG', 'ST_Slope']] = data[['ChestPainType', 'RestingECG', 'ST_Slope']].apply(encoder.fit_transform)

data = pd.get_dummies(data, columns = ['Sex', 'ExerciseAngina'])
data

# In[17]:

st.write('Confirming the Absence of Null values')
data.isna().sum()


st.write('Data appears clean now') 

st.markdown('EDA QUESTIONS AND ANSWERS')

st.write('Since the HeartDisease is the output column, can we know the count of those who had Heart disease based on their Sex?')

f = data.groupby(['Sex_F', 'Sex_M'])['HeartDisease'].count()
f
st.write('This shows that 725 men were diagnosed with heart disease while only 193 women had the same disease in a sample of 918 people')

st.markdown('Can we find any underaged with Heart disease in this sample data?')

# In[19]:


g = data.groupby(data['Age']<18).count()['HeartDisease']
g
st.write('This shows that no individual below the age of 18 who had the disease')


st.markdown('DATA VISUALIZATIONS')

# In[20]:


import seaborn as sns


# In[21]:


# finding the relationship between the features
st.title('Correllation Graph')
fig2 = plt.figure()
corr = data.corr()                              
sns.heatmap(corr)
st.pyplot(fig2)
st.write('the warmer the color, the higher the correllation.')
# #### There appears to be no strong relationship between the variables.


# In [29]:
st.title('Boxplot Representation of Heart Disease against age')
fig3 = plt.figure()
sns.boxplot(x="HeartDisease", y="Age", data =data)
st.pyplot(fig3)
st.write('We know from this that although no underaged was diagnosed of the disease. there are a few under 40s (between 30 and 35 years) who were diagnosed. They were the outliers. The median age of those who were diagnosed is a little above 60 years.')

st.title('Barplot Representation of Blood Sugar')
fig5 = plt.figure()
sns.barplot(x="FastingBS", y="ST_Slope", hue= 'RestingECG', data=data)
st.pyplot(fig5)


st.title('Model Prediction')


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier


# In[38]:


X= data.drop(['HeartDisease'], axis = 1).values
y= data['HeartDisease'].values              


# # Using Logistic Regression

# In[39]:


# split to 70% train dataset and 30% test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 45)


# In[40]:


clf = LogisticRegression()


# In[41]:


clf.fit(X_train, y_train)
Ypred = clf.predict(X_test)


# In[42]:


Ypred


# In[43]:


predictions = pd.DataFrame(Ypred).rename(columns= {0: 'predictions'})
predictions


# In[44]:


from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))


# # Using Cross Validation

# In[45]:

st.title('Cross Validation')
model = KNeighborsClassifier()
# evaluate the model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=10, n_jobs= -1)
n_scores


# In[46]:


st.write("Cross Validation Scores: ", n_scores)

st.write("Average CV Score: ", n_scores.mean())

st.write("Number of CV Scores used in Average: ", len(n_scores))


# In[47]:


from sklearn import metrics


# In[48]:

st.title('Confusion matrix')
confusion_matrix = metrics.confusion_matrix(y_test, predictions)
fig = plt.figure()
sns.heatmap(confusion_matrix, annot = True)
st.pyplot(fig)


st.write('From this matrix, out of 256 predictions, we had 77 True Negatives and 79 True Positives. 28 datapoints were predicted wrongly.')

# In[ ]:




