#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/duonghung86/Injury-severity-classification/blob/main/VCA_2_1_MLP_resampling.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[2]:


from psutil import virtual_memory
ram_gb = virtual_memory().total / 1e9
print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))

if ram_gb < 20:
  print('To enable a high-RAM runtime, select the Runtime > "Change runtime type"')
  print('menu, and then select High-RAM in the Runtime shape dropdown. Then, ')
  print('re-execute this cell.')
else:
  print('You are using a high-RAM runtime!')


# In[3]:


import warnings
warnings.filterwarnings("ignore")


# In[4]:


#!cat /proc/cpuinfo


# In[5]:


import psutil
# gives a single float value
print(psutil.cpu_percent())
# gives an object with many fields
psutil.virtual_memory()


# In[19]:


# Basic packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time
import os
# Preprocessing
from sklearn.preprocessing import StandardScaler # Standardization
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Machine learning algos
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from collections import Counter
from imblearn.over_sampling import SMOTE, RandomOverSampler,BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler,NearMiss,EditedNearestNeighbours
# Metrics
from imblearn.metrics import geometric_mean_score
# Tensorflow
import tensorflow as tf
print(tf.__version__)
from tensorflow import feature_column  # for data wrangling
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping


# In[7]:


# Download dataset
url = 'https://github.com/duonghung86/Injury-severity-classification/blob/main/Prepared%20Texas%202019.zip?raw=true' 
data_path = tf.keras.utils.get_file(origin=url, fname=url.split('/')[-1].split('?')[0], extract=True)
data_path = data_path.replace('%20',' ').replace('.zip','.csv')


# In[8]:


# Load data
df = pd.read_csv(data_path)
print(df.shape)
df.head(3)


# In[9]:


# Let's just use 80% of the total dataset
#df, _ = train_test_split(df, test_size=0.9,stratify = df['Prsn_Injry_Sev'])
#df.shape


# In[10]:


y = df['Prsn_Injry_Sev']
print('All target values:')
print(y.value_counts())
X = df.drop(columns=['Prsn_Injry_Sev'])


# In[11]:


# %% Data wrangling -------------
# Classify variable type
emb_vars, ind_vars, num_vars = [], [], []
for var in X.columns:
    if X[var].dtypes == 'O':
        if len(X[var].unique()) > 5:
            emb_vars.append(var)
        else:
            ind_vars.append(var)
    else:
        num_vars.append(var)
print('Numerical variables are ', num_vars)
print('Categorical variables that have at most 5 categories are ', ind_vars)
print('Categorical variables that have more than 5 categories are ', emb_vars)

# Create feature columns
feature_columns = []
# numeric cols
for header in num_vars:
    feature_columns.append(feature_column.numeric_column(header))
# bucketized cols
# age = feature_column.numeric_column('Prsn_Age')
# age_buckets = feature_column.bucketized_column(age, boundaries=[16, 22, 35, 55, 65])
# feature_columns.append(age_buckets)
# indicator_columns
for col_name in ind_vars:
    categorical_column = feature_column.categorical_column_with_vocabulary_list(
        col_name, X[col_name].unique())
    indicator_column = feature_column.indicator_column(categorical_column)
    feature_columns.append(indicator_column)
# embedding columns
for col_name in emb_vars:
    emb_column = feature_column.categorical_column_with_vocabulary_list(
        col_name, X[col_name].unique())
    col_embedding = feature_column.embedding_column(emb_column, dimension=5)
    feature_columns.append(col_embedding)

# Convert all setup into new dataset
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
X = feature_layer(dict(X)).numpy()
print('New shape of the input data set:',X.shape)


# In[12]:


# %% Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=48)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=48)

print('Training features shape:', X_train.shape)
print('Validation features shape:', X_val.shape)
print('Test features shape:', X_test.shape)

# %% standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


# In[22]:


# Import Metrics
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import roc_auc_score,confusion_matrix,accuracy_score

# %% Function to compare the prediction and true labels
def get_accs2(label, pred_proba, tr_time=0,index=None):
    prediction = pred_proba.argmax(axis=1)
    cm = confusion_matrix(label, prediction)
    length = cm.shape[0]
    num_cases = len(label)
    # global accuracy
    glb_acc = np.trace(cm) / len(label)
    ind_accs = cm / np.sum(cm, axis=1)[:, np.newaxis]
    accs = [ind_accs[i, i] for i in range(length)]
    cols = ['Class {}'.format(i) for i in range(length)]
    # Global accuracy
    accs.append(glb_acc)
    # AUC
    accs.append(roc_auc_score(label, pred_proba,multi_class='ovr'))
    # G-mean
    accs.append(geometric_mean_score(label, prediction, correction=0.001))
    # Average perf
    accs.append(np.mean(accs[-3:]))
    # Training time
    accs.append(np.round(tr_time,3))
    cols = cols + ['Accuracy','AUC','G-mean','Avg_Pfm','Training Time']

    out = np.array(accs).reshape(1, len(accs))
    return pd.DataFrame(out, columns=cols,index=[index])


# In[13]:


early_stop = {'roc_auc':       tf.keras.metrics.AUC(name='roc_auc',curve='ROC'),
               'pr_auc':       tf.keras.metrics.AUC(name='pr_auc',curve='PR')                    
               }


# In[14]:


def create_mlp(metric):
    MLP = Sequential([Dense(10,
                           activation='relu',
                           input_dim=X_train.shape[1]
                           ),
                      Dropout(0.5),
                      Dense(5, activation='softmax')])
    MLP.compile(optimizer='adam',
                #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                metrics=[metric]
               )
    return MLP


# In[15]:


def early_stops(metric_name):
    es = EarlyStopping(monitor='val_'+ metric_name,
                   verbose=1,
                   patience=10,
                   mode='max',
                   restore_best_weights=True)
    return es


# # Normal Resampling

# In[38]:


EPOCH = 50


# In[39]:


Resamples = {'ROS':             RandomOverSampler(), 
             }
rsts = pd.DataFrame()
for key,value in Resamples.items():
    start = time.time()
    X_res, y_res = value.fit_resample(X_train, y_train)
    end = time.time()
    res_time = end - start
    print(key)
    print('Resampled dataset shape %s' % Counter(y_res))
    print('resampling time is {0:.2f} seconds'.format(res_time))
    for name, metric in early_stop.items():
        model = create_mlp(metric)
        start = time.time()
        monitor = model.fit(X_res, pd.get_dummies(y_res).values,
                        callbacks=[early_stops(name)],
                        validation_data=(X_val, pd.get_dummies(y_val).values),
                        verbose=0, epochs = EPOCH)
        end = time.time()
        # use the model to make predictions with the test data
        y_pred = model.predict(X_test)
        # get the evaluation metrics
        result = get_accs2(y_test.values, y_pred, end-start,'MLP-'+key+'-'+name)
        result['Res_time'] = res_time
        rsts = rsts.append(result)
rsts.to_csv('MLP_AUC_RES '+time.ctime()+'.csv')
print(rsts.iloc[:,5:])


# # Hybrid Resampling
# 

# In[40]:


y_dict = Counter(y_train)
y_dict


# ## Oversampling and then undersampling

# In[41]:


ss = {}
for i in range(2,5):
    ss[i] = y_dict[1]
oses = ['ROS','SMOTE']
uses = ['RUS']


# In[42]:


for os_name in oses:
    for us_name in uses:
        key = os_name + '-'+us_name
        print(key)
        if us_name == 'ROS': 
            res = RandomOverSampler(sampling_strategy=ss)
        else: res = SMOTE(sampling_strategy=ss)
        start = time.time()
        X_res, y_res = res.fit_resample(X_train, y_train)

        if us_name == 'RUS': 
            res = RandomUnderSampler(sampling_strategy='majority')
        else: res = NearMiss(sampling_strategy='majority')  
        X_res, y_res = res.fit_resample(X_res, y_res)
        end = time.time()
        res_time = end-start

        print('Resampled dataset shape %s' % Counter(y_res))
        print('Resamling time %.2f sec' % (res_time))
        for name, metric in early_stop.items():
            model = create_mlp(metric)
            start = time.time()
            monitor = model.fit(X_res, pd.get_dummies(y_res).values,
                            callbacks=[early_stops(name)],
                            validation_data=(X_val, pd.get_dummies(y_val).values),
                            verbose=0, epochs = EPOCH)
            end = time.time()
            # use the model to make predictions with the test data
            y_pred = model.predict(X_test)
            # get the evaluation metrics
            result = get_accs2(y_test.values, y_pred, end-start,'MLP-'+key+'-'+name)
            result['Res_time'] = res_time
            rsts = rsts.append(result)
rsts.to_csv('MLP_AUC_RES '+time.ctime()+'.csv')


# ## Under sampling and then over sampling

# In[43]:


oses = ['ROS','SMOTE']
uses = ['RUS','NearMiss']


# In[44]:


for us_name in uses:
    for os_name in oses:
        key = us_name + '2'+os_name
        print(key)
        if us_name == 'RUS': 
            res = RandomUnderSampler(sampling_strategy={0: y_dict[1]})
        else: res = NearMiss(sampling_strategy={0: y_dict[1]})
        start = time.time()
        X_res, y_res = res.fit_resample(X_train, y_train)

        if us_name == 'ROS': 
            res = RandomOverSampler(sampling_strategy='not majority')
        else: res = SMOTE(sampling_strategy='not majority')
        X_res, y_res = res.fit_resample(X_res, y_res)
        end = time.time()
        res_time = end-start
        print('Resampled dataset shape %s' % Counter(y_res))
        print('Resamling time %.2f sec' % (res_time))
        for name, metric in early_stop.items():
            model = create_mlp(metric)
            start = time.time()
            monitor = model.fit(X_res, pd.get_dummies(y_res).values,
                            callbacks=[early_stops(name)],
                            validation_data=(X_val, pd.get_dummies(y_val).values),
                            verbose=0, epochs = EPOCH)
            end = time.time()
            # use the model to make predictions with the test data
            y_pred = model.predict(X_test)
            # get the evaluation metrics
            result = get_accs2(y_test.values, y_pred, end-start,'MLP-'+key+'-'+name)
            result['Res_time'] = res_time
            rsts = rsts.append(result)
rsts.to_csv('MLP_AUC_RES '+time.ctime()+'.csv')


# In[49]:


time.ctime()


# In[50]:


rsts.to_csv('MLP_AUC_RES '+time.ctime()+'.csv')

