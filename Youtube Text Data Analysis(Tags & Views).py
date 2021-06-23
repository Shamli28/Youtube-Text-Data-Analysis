#!/usr/bin/env python
# coding: utf-8

# ### Analyze Tags column, what are trending tags on youtube

# In[1]:


# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


videos=pd.read_csv('/home/shamli/Downloads/Datasets/1-Youtube Text Data Analysis/USvideos.csv', error_bad_lines=False)


# In[3]:


videos.head()


# In[6]:


tags_complete=' '.join(videos['tags'])
tags_complete


# In[11]:


import re #regular expression model


# In[12]:


#  whenever u have to deal with some kind of text data from this re you have to use substitute functn
tags=re.sub('[^a-zA-Z]',' ', tags_complete)


# In[13]:


tags


# In[15]:


tags=re.sub(' +',' ',tags) # to denote your space,spacing plus so that i have extra space just remove it with one space 


# In[17]:


from wordcloud import WordCloud,STOPWORDS  #he,she,it,they,that(they make nosense at all)


# In[18]:


wordcloud=WordCloud(width=1000,height=500,stopwords=set(STOPWORDS)).generate(tags)


# In[20]:


plt.figure(figsize=(15,5))
plt.imshow(wordcloud)
plt.axis('off')


# ### Perform analysis on Likes,Views & Dislikes & find how they are co-related with each other

# - Used regression plot,because whenever you have to find some kind of relation between two variables,regression plot will be very handy 

# In[22]:


sns.regplot(data=videos,x='views',y='likes')
plt.title('Regression plot for Views & Likes')


# - Here,Views are increasing,Likes are also increasing

# In[24]:


sns.regplot(data=videos,x='views',y='dislikes')
plt.title('Regression plot for Views & disLikes')


# - Here, Views are increasing, but dislikes are not

# In[25]:


df_corr=videos[['views','likes','dislikes']] # features we want


# In[27]:


df_corr.corr()


# In[29]:


sns.heatmap(df_corr.corr(),annot=True)   # have to make it user friendly/smooth,pass the annot parameter


# - Views & Likes has highest correlation,because likes vs likes & dislike vs dislike it make no sense.

# In[ ]:




