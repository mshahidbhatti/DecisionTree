#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Tutorial
# 

# # Decision Tree

# ![simple%20data.png](attachment:simple%20data.png)

# ![Noseparable%20data.png](attachment:Noseparable%20data.png)

# ![DecisionTree.png](attachment:DecisionTree.png)
# 

# ![tree1.png](attachment:tree1.png)

# ![tree3.png](attachment:tree3.png)

# ![tree2.png](attachment:tree2.png)

# In[149]:


import pandas as pd

data = pd.read_csv("C:/Users/mshah/OneDrive/Desktop/data.csv")

data.head()


# In[162]:


from sklearn.preprocessing import LabelEncoder

le_playtennis = LabelEncoder()




# In[163]:



inputs['Outlook_n']=le_playtennis.fit_transform(data['Outlook'])
inputs['Temperature_n']=le_playtennis.fit_transform(data['Temperature'])
inputs['Humidity_n']=le_playtennis.fit_transform(data['Humidity'])
inputs['Wind_n']=le_playtennis.fit_transform(data['Wind'])
inputs['PlayTennis_n'] = le_playtennis.fit_transform(data['PlayTennis'])


# In[164]:


inputs


# In[153]:


target=inputs['PlayTennis_n']
target


# In[154]:


inputs = inputs.drop(['PlayTennis_n'], axis='columns')


# In[155]:


inputs


# In[156]:


from sklearn import tree


# In[157]:


model = tree.DecisionTreeClassifier(criterion='entropy')


# In[158]:


model.fit(inputs,target)


# In[159]:


model.score(inputs,target)


# In[160]:


model.predict([[2,2,1,0]])

