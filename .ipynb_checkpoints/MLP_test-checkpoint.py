from psutil import virtual_memory,cpu_percent
ram_gb = virtual_memory().total / 1e9
print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))
print('Current system-wide CPU utilization %: ',cpu_percent())
#Remove all warning
import warnings
warnings.filterwarnings("ignore")


# In[3]:


# Basic packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time
import os
#INFO and WARNING messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
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


# # Load data

# In[4]:


url = 'https://github.com/duonghung86/Injury-severity-classification/blob/main/Prepared%20Texas%202019.zip?raw=true' 
data_path = tf.keras.utils.get_file(origin=url, fname=url.split('/')[-1].split('?')[0], extract=True)
data_path = data_path.replace('%20',' ').replace('.zip','.csv')


# In[5]:


# Load data
df = pd.read_csv(data_path)
print(df.shape)
df.head(3)


# In[6]:


# Let's just use 80% of the total dataset
df, _ = train_test_split(df, test_size=0.50,stratify = df['Prsn_Injry_Sev'])
print(df.shape)


# In[7]:


y = df['Prsn_Injry_Sev']
print('All target values:')
print(y.value_counts())
X = df.drop(columns=['Prsn_Injry_Sev'])


# In[8]:


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


# In[9]:


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


# In[10]:


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



rsts = pd.DataFrame()

# Add weights
weights = len(y) / (5 * np.bincount(y))
cls_wgt = dict(zip(np.arange(5), weights))
cls_wgt


# In[13]:


es = EarlyStopping(monitor='val_cohen_kappa',
                   verbose=1, patience=10, mode='max',
                   restore_best_weights=True)


# In[14]:


def create_mlp():
    MLP = Sequential([Dense(10,activation='relu',
                           input_dim=X_train.shape[1]),
                      Dropout(0.5),
                      Dense(5, activation='softmax')])
    MLP.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[CohenKappa(num_classes=5,sparse_labels=True)])
    return MLP


# In[15]:


model = create_mlp()
start = time.time()
monitor = model.fit(X_train, y_train.values,
                    callbacks=[es],
                    class_weight = cls_wgt,
                    validation_data=(X_val, y_val.values),
                    verbose=0, epochs=5)
end = time.time()
# use the model to make predictions with the test data
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
# get the evaluation metrics
result = get_accs(y_test.values, y_pred, end-start)
result.index = ['MLP-Weights']
rsts = rsts.append(result)
print(rsts.iloc[:,5:])