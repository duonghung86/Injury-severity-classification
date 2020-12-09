#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/duonghung86/Injury-severity-classification/blob/main/VCA_2_1_MLP_resampling.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[1]:


from psutil import virtual_memory,cpu_percent
ram_gb = virtual_memory().total / 1e9
print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))
print('Current system-wide CPU utilization %: ',cpu_percent())
#Remove all warning
import warnings
warnings.filterwarnings("ignore")


# In[2]:


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
from tensorflow_addons.metrics import CohenKappa,F1Score
from tensorboard.plugins.hparams import api as hp


# # Load data

# In[3]:


url = 'https://github.com/duonghung86/Injury-severity-classification/blob/main/Prepared%20Texas%202019.zip?raw=true' 
data_path = tf.keras.utils.get_file(origin=url, fname=url.split('/')[-1].split('?')[0], extract=True)
data_path = data_path.replace('%20',' ').replace('.zip','.csv')


# In[4]:


# Load data
df = pd.read_csv(data_path)
print(df.shape)
df.head(3)


# In[5]:


# Let's just use 80% of the total dataset
#df, _ = train_test_split(df, test_size=0.80,stratify = df['Prsn_Injry_Sev'])
#df.shape


# In[6]:


y = df['Prsn_Injry_Sev']
print('All target values:')
print(y.value_counts())
X = df.drop(columns=['Prsn_Injry_Sev'])


# In[7]:


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


# In[8]:


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


# In[9]:


# %% Function to compare the prediction and true labels
def get_accs(label, prediction, tr_time,  show=False):
    cm = confusion_matrix(label, prediction)
    length = cm.shape[0]
    num_cases = len(label)
    # global accuracy
    glb_acc = np.trace(cm) / len(label)
    ind_accs = cm / np.sum(cm, axis=1)[:, np.newaxis]
    accs = [ind_accs[i, i] for i in range(length)]
    index = ['Class {}'.format(i) for i in range(length)]
    # Global accuracy
    accs.append(glb_acc)
    #index.append
    # G-mean
    accs.append(geometric_mean_score(label, prediction, correction=0.001))
    #index.append('G-mean')
    # Average perf
    accs.append((glb_acc + accs[-1]) / 2)
    #index.append('Avg_Pfm')
    # Training time
    accs.append(np.round(tr_time,3))
    index = index + ['Overall Accuracy','G-mean','Avg_Pfm','Training Time']
    # Plot confusion matrix
    plot_dict = {'Confusion matrix': (cm,'g'),
                 'Normalized confusion matrix': (ind_accs,'.2f')}
    if show:
        plt.figure(figsize=(14, 6))
        i = 1
        for key, value in plot_dict.items():
            plt.subplot(1, 2, i)
            sns.heatmap(value[0], xticklabels=np.arange(length), yticklabels=np.arange(length),
                        annot=True, fmt=value[1], cmap="YlGnBu")
            plt.xlabel('Prediction')
            plt.ylabel('Label')
            plt.title(key)
            i+= 1
        plt.show()
    out = np.array(accs).reshape(1, len(accs))
    return pd.DataFrame(out, columns=index)


# # Class weights

# In[10]:


wgt='balanced'
clfs = [LogisticRegression(solver = 'lbfgs',class_weight=wgt),
        DecisionTreeClassifier(class_weight=wgt),
        RandomForestClassifier(max_depth=4,class_weight=wgt)]
clf_names = ['LR','DT','RF']
rsts = pd.DataFrame()
for model, name in zip(clfs,clf_names):
    start = time.time()
    print(name)
    model.fit(X_train, y_train.values)
    # use the model to make predictions with the test data
    y_pred = model.predict(X_test)
    end= time.time()
    # get the evaluation metrics
    result = get_accs(y_test.values, y_pred, end-start)
    result.index = [name + '-Weights']
    rsts = rsts.append(result)
print(rsts.iloc[:,5:])


# ## MLP

# In[11]:


# Add weights
weights = len(y) / (5 * np.bincount(y))
cls_wgt = dict(zip(np.arange(5), weights))
cls_wgt


# In[12]:


es = EarlyStopping(monitor='val_cohen_kappa',
                   verbose=1, patience=10, mode='max',
                   restore_best_weights=True)


# In[13]:


def create_mlp():
    MLP = Sequential([Dense(10,activation='relu',
                           input_dim=X_train.shape[1]),
                      Dropout(0.5),
                      Dense(5, activation='softmax')])
    MLP.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[CohenKappa(num_classes=5,sparse_labels=True)])
    return MLP


# In[14]:


model = create_mlp()
start = time.time()
monitor = model.fit(X_train, y_train.values,
                    callbacks=[es],
                    class_weight = cls_wgt,
                    validation_data=(X_val, y_val.values),
                    verbose=0, epochs=50)
end = time.time()
# use the model to make predictions with the test data
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
# get the evaluation metrics
result = get_accs(y_test.values, y_pred, end-start)
result.index = ['MLP-Weights']
rsts = rsts.append(result)
print(rsts.iloc[:,5:])


# # Normal Resampling

# In[15]:


Resamples = {'ROS':             RandomOverSampler(), 
             'SMOTE':           SMOTE(random_state=42),
             'BorderlineSMOTE': BorderlineSMOTE(),
             'RUS': RandomUnderSampler(), 
             'NearMiss': NearMiss()
             }
for key,value in Resamples.items():
    start = time.time()
    X_res, y_res = value.fit_resample(X_train, y_train)
    end = time.time()
    res_time = end - start
    print(key)
    print('Resampled dataset shape %s' % Counter(y_res))
    print('resampling time is {0:.2f} seconds'.format(res_time))
    
    #Logistic model
    LR = LogisticRegression(solver = 'lbfgs')
    start = time.time()
    LR.fit(X_res, y_res)
    end= time.time()
    # get the evaluation metrics
    # use the model to make predictions with the test data
    y_pred = LR.predict(X_test)
    result = get_accs(y_test.values,y_pred, tr_time= end-start)
    result['Resample time'] = res_time
    result.index = ['LR-' + key]
    rsts = rsts.append(result)
    
print(rsts.iloc[:,5:])


# In[16]:


#rsts = rsts.iloc[:9,:]
#rsts


# # Hybrid Resampling
# 

# In[17]:


y_dict = Counter(y_train)
y_dict


# ## Oversampling and then undersampling

# In[18]:


ss = {}
for i in range(2,5):
    ss[i] = y_dict[1]
oses = ['ROS','SMOTE','BorderlineSMOTE']
uses = ['RUS','NearMiss']


# In[19]:


for os_name in oses:
    for us_name in uses:
        print(os_name,'-',us_name)
        if os_name == 'ROS': 
            res = RandomOverSampler(sampling_strategy=ss)
        elif os_name == 'BorderlineSMOTE': 
            res = BorderlineSMOTE(sampling_strategy=ss)
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
        LR = LogisticRegression(solver = 'lbfgs')
        start = time.time()
        LR.fit(X_res, y_res)
        end= time.time()
        # get the evaluation metrics
        # use the model to make predictions with the test data
        y_pred = LR.predict(X_test)
        result = get_accs(y_test.values,y_pred, tr_time= end-start)
        result['Resample time'] = res_time
        result.index = ['LR-' + os_name + '-' + us_name]
        rsts = rsts.append(result)

print(rsts.iloc[:,5:])


# ## Under sampling and then over sampling

# In[20]:


for us_name in uses:
    for os_name in oses:
        print(us_name,'-',os_name)
        if us_name == 'RUS': 
            res = RandomUnderSampler(sampling_strategy={0: y_dict[1]})
        else: res = NearMiss(sampling_strategy={0: y_dict[1]})
        start = time.time()
        X_res, y_res = res.fit_resample(X_train, y_train)

        if os_name == 'ROS': 
            res = RandomOverSampler(sampling_strategy='not majority')
        elif os_name == 'BorderlineSMOTE': 
            res = BorderlineSMOTE(sampling_strategy='not majority')
        else: res = SMOTE(sampling_strategy='not majority')
        X_res, y_res = res.fit_resample(X_res, y_res)
        end = time.time()
        res_time = end-start
        print('Resampled dataset shape %s' % Counter(y_res))
        print('Resamling time %.2f sec' % (res_time))
        LR = LogisticRegression(solver = 'lbfgs',
                                #class_weight= 'balanced'
                                )
        start = time.time()
        LR.fit(X_res, y_res)
        end= time.time()
        # get the evaluation metrics
        # use the model to make predictions with the test data
        y_pred = LR.predict(X_test)
        result = get_accs(y_test.values,y_pred, tr_time= end-start)
        result['Resample time'] = res_time
        result.index = ['LR-' + us_name + '-' + os_name]
        rsts = rsts.append(result)

print(rsts.iloc[:,5:])


# ### Refining result
# NearMiss not only increased the computing time but also reduced the accuracy significantly
# Apply 5-fold cross validation
# 

# In[21]:


# Border → RUS
bor2rus = pd.DataFrame()
for i in range():
    print(i)
    start = time.time()
    res = BorderlineSMOTE(sampling_strategy=ss)
    print('sampling #1 ...')
    X_res, y_res = res.fit_resample(X_train, y_train)
    res = RandomUnderSampler(sampling_strategy='majority')
    print('sampling #2 ...')
    X_res, y_res = res.fit_resample(X_res, y_res)
    end = time.time()
    res_time = end-start
    # training and prediction
    start = time.time()
    LR = LogisticRegression(solver = 'lbfgs',
                            #class_weight= 'balanced'
                            )
    start = time.time()
    print('training...')
    LR.fit(X_res, y_res)
    end= time.time()
    # get the evaluation metrics
    # use the model to make predictions with the test data
    y_pred = LR.predict(X_test)
    result = get_accs(y_test.values,y_pred, tr_time= end-start)
    result['Resample time'] = res_time
    bor2rus = bor2rus.append(result)
bor2rus


# In[22]:


bor2rus.describe().iloc[1:3,5:]


# In[23]:


# RUS → Borderline
rus2bor = pd.DataFrame()
for i in range(5):
    print(i)
    start = time.time()
    res = RandomUnderSampler(sampling_strategy={0: y_dict[1]})
    print('sampling #1 ...')
    X_res, y_res = res.fit_resample(X_train, y_train)
    res = BorderlineSMOTE(sampling_strategy='not majority')
    print('sampling #2 ...')
    X_res, y_res = res.fit_resample(X_res, y_res)
    end = time.time()
    res_time = end-start
    # training and prediction
    start = time.time()
    LR = LogisticRegression(solver = 'lbfgs',
                            #class_weight= 'balanced'
                            )
    start = time.time()
    print('training...')
    LR.fit(X_res, y_res)
    end= time.time()
    # get the evaluation metrics
    # use the model to make predictions with the test data
    y_pred = LR.predict(X_test)
    result = get_accs(y_test.values,y_pred, tr_time= end-start)
    result['Resample time'] = res_time
    rus2bor = rus2bor.append(result)
rus2bor


# In[1]:


print(rus2bor.describe().iloc[1:3,5:])
print(bor2rus.describe().iloc[1:3,5:])


# In[29]:


start = time.time()
res = RandomUnderSampler(sampling_strategy={0: y_dict[1]})
print('sampling #1 ...')
X_res, y_res = res.fit_resample(X_train, y_train)
res = BorderlineSMOTE(sampling_strategy='not majority')
print('sampling #2 ...')
X_res, y_res = res.fit_resample(X_res, y_res)
print(Counter(y_res))
end = time.time()
res_time = end-start
model = create_mlp()
start = time.time()
model.fit(X_res, y_res,
                    callbacks=[es],
                 #   class_weight = cls_wgt,
                    validation_data=(X_val, y_val.values),
                    verbose=1, epochs=50)
end = time.time()
# use the model to make predictions with the test data
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
# get the evaluation metrics
result = get_accs(y_test.values, y_pred, end-start)
result['Resample time'] = res_time
result.index = ['MLP-RUS2BOR']
rsts = rsts.append(result)
print(rsts.iloc[:,5:])


# In[30]:


rsts.to_csv('VCA_Resamplings.csv')

