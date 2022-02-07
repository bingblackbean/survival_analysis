#!/usr/bin/env python
# coding: utf-8

# # 模型选择
# 
# 前面介绍了多种建模方式，这节我们讨论以下如何选择模型。

# In[1]:


import pandas as pd
# read training data 
train_df = pd.read_csv('../data/PM_train.txt', sep=" ", header=None)
train_df.drop(train_df.columns[[26, 27]], axis=1, inplace=True)
train_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                     's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                     's15', 's16', 's17', 's18', 's19', 's20', 's21']
# Data Labeling - generate column RUL
rul = pd.DataFrame(train_df.groupby('id')['cycle'].max()).reset_index()
rul.columns = ['id', 'max']
train_df = train_df.merge(rul, on=['id'], how='left')
train_df['observed'] = (train_df['cycle'] >= train_df['max'])

import lifelines
from matplotlib import pyplot as plt
max_event = train_df.groupby('id').last()
max_event_censor_1 = train_df[train_df['max']<=250].groupby('id').last()  
max_event_censor_2 = train_df[(train_df['max']>250) & (train_df['cycle']>250)].groupby('id').sample(n=1,random_state=21)

max_event_censor_mix = pd.concat([max_event_censor_1,max_event_censor_2])


# ## 检查低方差
# 我们可以使用DataFrame的统计值,筛选出出方差比较大的特征

# In[2]:


std_sr = train_df.drop(['cycle','id','max'],axis=1).describe().loc['std'].sort_values(ascending=False)
std_sr


# In[3]:


list(std_sr.iloc[0:5].index)


# In[4]:


from lifelines import CoxPHFitter,WeibullAFTFitter

cph1 = CoxPHFitter(label='cph1')
cph1.fit(df=max_event_censor_mix[list(std_sr.iloc[0:5].index)+['cycle','observed']],duration_col='cycle',
event_col='observed',show_progress=True)


# In[5]:


cph2 = CoxPHFitter(label='cph1')
cph2.fit(df=max_event_censor_mix[list(std_sr.iloc[0:10].index)+['cycle','observed']],duration_col='cycle',
event_col='observed',show_progress=True)


# In[6]:


cph3 = CoxPHFitter(label='cph1')
cph3.fit(df=max_event_censor_mix[list(std_sr.iloc[0:15].index)+['cycle','observed']],duration_col='cycle',
event_col='observed',show_progress=True)


# 我们已经学了多个模型,现在是时候做出抉择了。
# 
# 其实对于一般的机器学习，我们会看模型的准确度。对于我们拟合的这个模型，他们同样有评价指标：对数似然。
# 
# 最大化似然性等同于最大化对数似然性，对数似然也是越大越好。可以看到top 15的那一组会有优势。
# 

# In[7]:


cph1_core = cph1.score(max_event_censor_mix[list(std_sr.iloc[0:15].index)+['cycle','observed']])
cph2_core = cph2.score(max_event_censor_mix[list(std_sr.iloc[0:15].index)+['cycle','observed']])
cph3_core = cph3.score(max_event_censor_mix[list(std_sr.iloc[0:15].index)+['cycle','observed']])
print(f'top 5 features, log likehood score is {cph1_core}')
print(f'top 10 features, log likehood score is is {cph2_core}')
print(f'top 15 features, log likehood score is is {cph3_core}')


# 但是，会不会有过拟合的风险呢？ 对于统计来说，主要参考一个指标AIC。
# AIC(model)=−2ll+2k
# 
# 其中k是模型的参数数量（自由度）和ll是最大对数似然。其实，AIC就是在使用尽可能少的参数最大化对数似然之间进行权衡。
# AIC_是参数模型的属性，对于 Cox 模型来说,AIC_partial_

# In[8]:


print(f'top 5 features, AIC is {cph1.AIC_partial_}')
print(f'top 10 features, AIC is {cph2.AIC_partial_}')
print(f'top 15 features, AIC is {cph3.AIC_partial_}')


# 我们会选择模型2。
# 最后，对比三个模型对同一个协变量的表现，发现总自由度（协变量个数）越少，单个协变量的影响越大（越激进）。

# In[9]:


ax = cph1.plot_partial_effects_on_outcome(['s9'], values=max_event_censor_mix[['s9']].quantile([0.1, 0.5,0.8]).values)
plt.xlabel('time')
plt.title('top 5 features')
plt.show()


# In[10]:


cph2.plot_partial_effects_on_outcome(['s9'], values=max_event_censor_mix[['s9']].quantile([0.1, 0.5,0.8]).values)

plt.xlabel('time')
plt.title('top 10 features')
plt.show()


# In[11]:


cph3.plot_partial_effects_on_outcome(['s9'], values=max_event_censor_mix[['s9']].quantile([0.1, 0.5,0.8]).values)

plt.xlabel('time')
plt.title('top 15 features')
plt.show()


# ## 总结
# 
# 本文介绍了模型协变量（特征)如何通过标准差筛选，以及如何通过AIC选择模型。

# In[ ]:




