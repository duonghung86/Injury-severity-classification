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
# Preprocessing
from sklearn.preprocessing import StandardScaler # Standardization
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from collections import Counter
from imblearn.over_sampling import SMOTE, RandomOverSampler,BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler,NearMiss,EditedNearestNeighbours
# Machine learning algos
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
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

#Remove all warning
import warnings
warnings.filterwarnings("ignore")


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
#df, _ = train_test_split(df, test_size=0.5,stratify = df['Prsn_Injry_Sev'])
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


y_dict = Counter(y_train)
start = time.time()
res = RandomUnderSampler(sampling_strategy={0: y_dict[1]})
print('sampling #1 ...')
X_train, y_train = res.fit_resample(X_train, y_train)
res = SMOTE(sampling_strategy='not majority')
print('sampling #2 ...')
X_train, y_train = res.fit_resample(X_train, y_train)
end = time.time()
res_time = end-start
print(Counter(y_train))
print(res_time)


# # ALL mini functions
# 
# 

# In[10]:


# %% Function to compare the prediction and true labels
def get_accs(label, prediction, show=False):
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
    index.append('Overall Accuracy')
    # G-mean
    accs.append(geometric_mean_score(label, prediction, correction=0.001))
    index.append('G-mean')
    # Average perf
    accs.append((glb_acc + accs[-1]) / 2)
    index.append('Avg_Pfm')
    if show:
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        sns.heatmap(cm, xticklabels=np.arange(length), yticklabels=np.arange(length),
                    annot=True, fmt='g', cmap="YlGnBu")
        plt.xlabel('Prediction')
        plt.ylabel('Label')
        plt.title('Confusion matrix')
        plt.subplot(1, 2, 2)
        sns.heatmap(ind_accs * 100, xticklabels=np.arange(length), yticklabels=np.arange(length),
                    annot=True, fmt='.2f', cmap="YlGnBu")
        plt.xlabel('Prediction')
        plt.ylabel('Label')
        plt.title('Normalized confusion matrix (%)')
        plt.show()
    out = np.array(accs).reshape(1, len(accs))
    return pd.DataFrame(out, columns=index)

def show_evolution(moni):
    hist = pd.DataFrame(monitor.history)
    no_metrics = np.int(hist.shape[1]/2)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 2), dpi=150)
    for i in range(2):
      hist.iloc[:,[i,no_metrics+i]].plot(ax=axes[i])
    plt.show()
# %% Produce an evaluation on the MLP model
def evaluation(model, monitor, time, name):
    # use the model to make predictions with the test data
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    # Show evolution of the training process
    # show_evolution(monitor)
    # get the evaluation metrics
    result = get_accs(y_test.values, y_pred)
    result['Training Time'] = np.round(time, 3)
    result.index = [name]
    return result


# In[11]:


wgt=None
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
    result = get_accs(y_test.values,y_pred)
    result['Training Time'] = np.round(end-start,3)
    result.index = [name]
    rsts = rsts.append(result)
print(rsts.iloc[:,-4:])


# # MLP functions
# 

# In[12]:


def early_stops(metric_name):
    es = EarlyStopping(monitor='val_'+ metric_name,
                   verbose=1,
                   patience=10,
                   mode='max',
                   restore_best_weights=True)
    return es


# # Ordinal multiclass
# 

# In[14]:


early_stop = {'accuracy':  tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
              'cohen_kappa': CohenKappa(num_classes=5,
                                        #sparse_labels=True
                                       ),
              'f1_micro': F1Score(num_classes=5,average="micro",threshold=0.5, name='f1_micro'),
               }


# In[15]:


def create_mlp(metric):
    MLP = Sequential([Dense(10,
                           activation='relu',
                           input_dim=X_train.shape[1]
                           ),
                      Dropout(0.5),
                      Dense(5, activation='softmax')])
    MLP.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                #loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                metrics=[metric]
               )
    return MLP


# In[16]:


for name, metric in early_stop.items():
    print(name)
    model = create_mlp(metric)
    start = time.time()
    monitor = model.fit(X_train, y_train,
                    callbacks=[early_stops(name)],
                    validation_data=(X_val, y_val.values),
                    verbose=1, epochs=50)
    end = time.time()
    rsts = rsts.append(evaluation(model, monitor, end - start,  'MLP '+ name))
    print(rsts.iloc[:,-4:])


# # One-hot encoded multiclass
# 

# In[ ]:


early_stop = {'auc':       tf.keras.metrics.AUC(name='auc'),
    'accuracy':  tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
              'precision': tf.keras.metrics.Precision(name='precision'),
              'recall':    tf.keras.metrics.Recall(name='recall'),
              
              }


# In[ ]:


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


# In[ ]:


#rsts = rsts.iloc[:6,:].copy()


# In[ ]:


for name, metric in early_stop.items():
    print(name)
    model = create_mlp(metric)
    start = time.time()
    monitor = model.fit(X_train, pd.get_dummies(y_train).values,
                    callbacks=[early_stops(name)],
                    validation_data=(X_val, pd.get_dummies(y_val).values),
                    verbose=0, epochs=50)
    end = time.time()
    rsts = rsts.append(evaluation(model, monitor, end - start,  'MLP 1H '+ name))
print(rsts.iloc[:,-4:])


# In[ ]:


rsts.to_csv('VCA_Res_ES_SMOTE.csv')

