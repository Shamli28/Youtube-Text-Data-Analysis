#!/usr/bin/env python
# coding: utf-8

# ### Performing Sentiment Analysis on Youtube Comments

# In[47]:


# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[48]:


comments = pd.read_csv('/home/shamli/Downloads/Datasets/1-Youtube Text Data Analysis/GBcomments.csv', error_bad_lines=False)


# - when we have parcel error we have to set error_bad_lines parameter to false

# In[49]:


comments.head()


# In[50]:


get_ipython().system('pip install textblob  # to perform text analysis we have to install external library')


# In[51]:


from textblob import TextBlob


# In[52]:


TextBlob('Its more accurate to call it the M+ (1000) be..').sentiment.polarity # have to pass te comment in textblob. what type of sentiment this comment has?


# - so call sentiment.polarity because you have to extract what exactly is the polarity of this one

# In[53]:


# Checking nUll values
comments.isna().sum()


# In[54]:


comments.dropna(inplace=True)   # to drop the missing value and set inplace parameter to true


# In[55]:


polarity=[]   # define a new list

for i in comments['comment_text']:
    polarity.append(TextBlob(i).sentiment.polarity) # access this Textblob & have to pass each n every i.& call this sentiment.polarity


# In[56]:


comments['polarity']=polarity   # define column in dataframe


# In[57]:


comments.head()


# ## WordCloud Representation of Sentiments

# ### Perform EDA for Positive Sentences

# In[58]:


comments_positive=comments[comments['polarity']==1] #pass the filter in data so i wl have my filter data from comments _positive


# In[59]:


comments_positive.shape


# In[60]:


comments_positive.head()


# - So,Lets say to visualize this comment_text feature, we will use WORDCLOUD. It's a tool,basically that type of tool,that type of functionality.
# - whenever you have to understand how imp our word is in some huge chunk of data.
# - So the more bigger word is,the more imp that word will be.

# In[61]:


get_ipython().system('pip install wordcloud')


# In[62]:


from wordcloud import WordCloud,STOPWORDS  #he,she,it,they,that(they make nosense at all)


# In[63]:


stopwords=set(STOPWORDS)     # set unique stopword


# In[64]:


total_comments=''.join(comments_positive['comment_text']) # store the entire data in existing data & perform join operation


# In[65]:


wordcloud=WordCloud(width=1000,height=500,stopwords=stopwords).generate(total_comments) # to genarate a workload call a generate fn


# In[66]:


plt.figure(figsize=(15,5))
plt.imshow(wordcloud) #show the wordcloud
plt.axis('off')# disable all the axis of wordclouds


# ## Perform EDA for Negative Sentence

# In[67]:


comments_negative=comments[comments['polarity']==-1]


# In[68]:


comments_negative.shape


# In[69]:


total_comments=''.join(comments_negative['comment_text'])


# In[70]:


wordcloud=WordCloud(width=1000,height=500,stopwords=stopwords).generate(total_comments)


# In[71]:


plt.figure(figsize=(15,5))
plt.imshow(wordcloud) 
plt.axis('off')


# ## Perform Emoji Analysis

# - Perform Emoji Analysis on comment

# In[72]:


comments.head()


# In[73]:


comments['comment_text'][1]


# In[74]:


print('\U0001F600')


# In[75]:


get_ipython().system('pip install emoji')


# In[76]:


import emoji


# In[85]:


len(comments)


# In[86]:


# extract emojis from this one text to 
comment=comments['comment_text'][1]


# In[88]:


[c for c in comment if c in emoji.UNICODE_EMOJI]


# In[89]:


str=''
for i in comments['comment_text']:
    list= [c for c in comment if c in emoji.UNICODE_EMOJI] #search each n every c & every character from each and every i
    for ele in list: # define new string
        str=str+ele  # simply concatenate each & every emoji in str


# In[81]:


len(str)


# In[82]:


str


# In[83]:


result=()
for i in set(str):
    result(i)=str.count(i)


# In[ ]:


result


# In[ ]:


result.items()


# In[ ]:


final=()
for key,value in sorted(result.items(),key =lambda item:item[1]):
    final[key]=value

