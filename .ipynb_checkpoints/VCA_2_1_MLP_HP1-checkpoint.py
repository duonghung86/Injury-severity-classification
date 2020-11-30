#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/duonghung86/Injury-severity-classification/blob/main/VCA_2_1_MLP_earlystopping.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

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
from collections import Counter
# Preprocessing
from sklearn.preprocessing import StandardScaler # Standardization
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Machine learning algos
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# Imblearn
from imblearn.metrics import geometric_mean_score
from imblearn.over_sampling import SMOTE, RandomOverSampler,BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler,NearMiss,EditedNearestNeighbours

# Grid search
from kerastuner.tuners import RandomSearch,Hyperband,BayesianOptimization
import kerastuner as kt
from tensorflow.keras.optimizers import Adam
# Tensorflow
import tensorflow as tf
print(tf.__version__)
from tensorflow import feature_column  # for data wrangling
from tensorflow.keras.losses import SparseCategoricalCrossentropy,CategoricalCrossentropy
from tensorflow.keras.layers import Dense,Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import SparseCategoricalAccuracy,CategoricalAccuracy
from tensorflow_addons.metrics import CohenKappa,F1Score


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
#df, _ = train_test_split(df, test_size=0.9,stratify = df['Prsn_Injry_Sev'])
df.shape


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


# # ALL mini functions
# 
# 

# In[9]:


# Import Metrics
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import roc_auc_score,confusion_matrix,accuracy_score

# %% Function to compare the prediction and true labels
def get_accs(label, pred_proba, tr_time=0,index=None):
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


# # ML with class weight

# # MLP functions
# 

# In[10]:


# Add weights
weights = len(y_train) / (5 * np.bincount(y_train))
cls_wgt = dict(zip(np.arange(5), weights))
cls_wgt


# In[11]:


def early_stops(metric_name):
    es = EarlyStopping(monitor='val_'+ metric_name,
                   verbose=1, patience=10, mode='max',
                   restore_best_weights=True)
    return es


# In[12]:


# Constant
EPOCH = 50
BATCH_SIZE = 2048
VERBOSE = 0


# In[13]:


METRICS = [SparseCategoricalAccuracy(name='accuracy'),
           CohenKappa(name='kappa',num_classes=5,sparse_labels=True),
           F1Score(name='f1_micro', num_classes=5,average="micro",threshold=0.5),
          ]
def create_mlp():
    MLP = Sequential([Dense(10,
                           activation='relu',
                           input_dim=X_train.shape[1],
                           ),
                      Dropout(0.5),
                      Dense(5, activation='softmax')])
    MLP.compile(optimizer='adam',
                loss=SparseCategoricalCrossentropy(from_logits=True),
                metrics=METRICS
               )
    return MLP


# # Hybrid Resampling 

# In[14]:


y_dict = Counter(y_train)

start = time.time()
res = RandomUnderSampler(random_state = 54, sampling_strategy={0: y_dict[1]})
print('under sampling ...')
X_res, y_res = res.fit_resample(X_train, y_train)

res = SMOTE(random_state = 34,sampling_strategy='not majority')
print(Counter(y_res))
print('over sampling #2 ...')
X_res, y_res = res.fit_resample(X_res, y_res)
end = time.time()
res_time = end-start
Counter(y_res),res_time


# # MLP with Hybrid Sampling

# In[15]:


rsts = pd.DataFrame()
for i in range(3):
    model = create_mlp()
    start = time.time()
    monitor = model.fit(X_res, y_res,
                        callbacks=[early_stops('accuracy')],
                        validation_data=(X_val,y_val),
                        batch_size=BATCH_SIZE,
                        verbose=VERBOSE, epochs=EPOCH
                       )
    end = time.time()
    # use the model to make predictions with the test data
    Y_pred = model.predict(X_test)
    rsts = rsts.append(get_accs(y_test.values,Y_pred,end-start,'MLP-HS#'+str(i+1)))
print(rsts.iloc[:,5:])


# # Set up grid search
# 
# We will investigates the following parameters:
# 
# - Initial weights
# - Activation function
# - Number of nodes
# - Dropout rate
# - Early Stop
# - Learning rate

# # Keras tuner

# In[16]:


def build_model(hp):
    hp_units = hp.Int('units', min_value=5, max_value=15, step=5)
    hp_learning_rate = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4]) 
    hp_dos = hp.Float('dropouts',min_value=0.2, max_value=0.5, step=0.1)
    hp_acts = hp.Choice('activation', values = ['relu','sigmoid','tanh','selu'])
    keins = ['uniform', 'lecun_uniform', 'normal', 'zeros', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
    hp_keins = hp.Choice('kernel_ini', values = keins) 
    model = Sequential([Dense(hp_units,
                           activation=hp_acts,
                           input_dim=X_train.shape[1],
                            kernel_initializer= hp_keins 
                           ),
                      Dropout(hp_dos),
                      Dense(5, activation='softmax')])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = hp_learning_rate),
                loss=SparseCategoricalCrossentropy(from_logits=True),
                metrics=METRICS
               )
    return model


# In[17]:


metr = ['loss','accuracy','kappa','f1_micro']


# In[ ]:


MAX_EPOCHS = 30
FACTOR = 5


# In[18]:


bps = pd.DataFrame()
for obj in metr:
    start = time.time()
    if obj in ['loss','accuracy']:
        goal = obj
    else:
        goal = kt.Objective('val_'+obj, direction="max")
    tuner = Hyperband(build_model,
                     objective = goal, 
                     max_epochs = MAX_EPOCHS,
                     factor = FACTOR,
                     directory = 'my_dir',
                     project_name = 'val_'+ obj+'_'+time.ctime())
    tuner.search(X_res, y_res,
                 epochs=MAX_EPOCHS,batch_size=2048,
                 verbose=0,
                 callbacks=[early_stops(obj)],
                 validation_data=(X_val, y_val))
    end = time.time()

    print('Tuning time is %.2f' % (end-start))
    print(tuner.oracle.get_best_trials(num_trials=1)[0].hyperparameters.values)
    bp = pd.Series(tuner.oracle.get_best_trials(num_trials=1)[0].hyperparameters.values,name=obj)
    #bp = bp.append(pd.Series(end-start,index=['Tuning_time']))
    bps = pd.concat((bps,bp),axis=1)
    models = tuner.get_best_models(num_models=FACTOR)
    for i in range(FACTOR):
        Y_pred = models[i].predict(X_test)
        rsts = rsts.append(get_accs(y_test.values,Y_pred,end-start,'MLP-'+obj+'-'+str(i+1)))
        rsts.to_csv('VCA_Tuning1.csv')
print(bps)
print(rsts.iloc[:,5:])
bps.to_csv('VCA_Tuning1_bps.csv')

