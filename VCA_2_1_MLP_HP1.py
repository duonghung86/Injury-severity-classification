#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/duonghung86/Injury-severity-classification/blob/main/VCA_2_1_MLP_earlystopping.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[2]:


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
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.optimizers import Adam
# Tensorflow
import tensorflow as tf
print(tf.__version__)
from tensorflow import feature_column  # for data wrangling
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.layers import Dense,Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow_addons.metrics import CohenKappa,F1Score
from tensorboard.plugins.hparams import api as hp
#Remove all warning
import warnings
warnings.filterwarnings("ignore")


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
#df, _ = train_test_split(df, test_size=0.9,stratify = df['Prsn_Injry_Sev'])
df.shape


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


# # ALL mini functions
# 
# 

# In[10]:


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


def create_mlp():
    MLP = Sequential([Dense(10,
                           activation='relu',
                           input_dim=X_train.shape[1],
                           ),
                      Dropout(0.5),
                      Dense(5, activation='softmax')])
    MLP.compile(optimizer='adam',
                loss=SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')]
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
                        validation_data=(X_val, y_val.values),
                        batch_size=BATCH_SIZE,
                        verbose=VERBOSE, epochs=EPOCH
                       )
    end = time.time()
    # use the model to make predictions with the test data
    Y_pred = model.predict(X_test)
    rsts = rsts.append(get_accs(y_test.values,Y_pred,end-start,'MLP-HS#'+str(i+1)))
rsts


# # Set up grid search
# 
# We will investigates the following parameters:
# 
# 1. Batch size
# 2. Epoch
# 3. Initial weights
# 4. Activation function
# 5. Number of nodes
# 6. Dropout rate
# 7. Early Stop
# 8. Learning rate

# # Keras Tuner

# ## Random Search

# In[46]:


from kerastuner.tuners import RandomSearch,Hyperband


# In[69]:


def build_model(hp):
    hp_units = hp.Int('units', min_value=5, max_value=15, step=5)
    hp_learning_rate = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4]) 
    hp_dos = hp.Float('dropouts',min_value=0.2, max_value=0.5, step=0.1)
    hp_acts = hp.Choice('activation', values = ['relu','sigmoid','tanh','selu'])
    keins = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
    hp_keins = hp.Choice('kernel_ini', values = keins) 
    metrics = ['accuracy',CohenKappa(num_classes=5,sparse_labels=True),
               F1Score(num_classes=5,average="micro",threshold=0.5),
              ]
    hp_metrics = hp.Choice('kernel_ini', values = keins)
    model = Sequential([Dense(hp_units,
                           activation=hp_acts,
                           input_dim=X_train.shape[1],
                            kernel_initializer= hp_keins 
                           ),
                      Dropout(hp_dos),
                      Dense(5, activation='softmax')])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = hp_learning_rate),
                loss=SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
               )
    return model


# In[71]:


tuner = Hyperband(build_model,
                     objective = 'val_loss', 
                     max_epochs = 30,
                     factor = 3,
                     directory = 'my_dir',
                     project_name = 'intro_to_kt')
tuner.search_space_summary()


# In[87]:


tuner.search(X_res, y_res,
             epochs=30,batch_size=1024,
             callbacks=[early_stops('loss')],
             validation_data=(X_val, y_val))


# In[73]:


tuner.results_summary()


# In[74]:


# Get the best parameters
tuner.oracle.get_best_trials(num_trials=1)[0].hyperparameters.values


# In[75]:


models = tuner.get_best_models(num_models=3)
Y_pred = models[0].predict(X_test)
get_accs(y_test.values,Y_pred,end-start,'MLP-HS#'+str(i+1))


# In[76]:


rsts = pd.DataFrame()
for i in range(3):
    Y_pred = models[i].predict(X_test)
    rsts = rsts.append(get_accs(y_test.values,Y_pred,end-start,'MLP-HS#'+str(i+1)))
rsts


# In[77]:


checkpoint_path = "training_1/cp.ckpt"
models[0].save_weights(checkpoint_path.format(epoch=0))


# In[84]:


# Test
def create_mlp():
    MLP = Sequential([Dense(10,
                           activation='relu',
                           input_dim=X_train.shape[1],
                           ),
                      Dropout(0.2),
                      Dense(5, activation='softmax')])
    MLP.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.01),
                loss=SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')]
               )
    return MLP
rsts = pd.DataFrame()
for i in range(3):
    model = create_mlp()
    model.load_weights(checkpoint_path)
    start = time.time()
    monitor = model.fit(X_res, y_res,
                        callbacks=[early_stops('loss')],
                        validation_data=(X_val, y_val.values),
                        batch_size=BATCH_SIZE,
                        verbose=VERBOSE, epochs=EPOCH
                       )
    end = time.time()
    # use the model to make predictions with the test data
    Y_pred = model.predict(X_test)
    rsts = rsts.append(get_accs(y_test.values,Y_pred,end-start,'MLP-HS#'+str(i+1)))
rsts


# In[85]:


# Add weights
weights = len(y_train) / (5 * np.bincount(y_train))
cls_wgt = dict(zip(np.arange(5), weights))
cls_wgt


# In[86]:


rsts = pd.DataFrame()
monitor = model.fit(X_train, y_train,
                        callbacks=[early_stops('loss')],
                        validation_data=(X_val, y_val.values),
                    class_weight=cls_wgt,
                        batch_size=BATCH_SIZE,
                        verbose=VERBOSE, epochs=EPOCH
                       )

# use the model to make predictions with the test data
Y_pred = model.predict(X_test)
rsts = rsts.append(get_accs(y_test.values,Y_pred,end-start,'MLP-HS#'+str(i+1)))
rsts


# ## Batch size

# In[ ]:


class MyTuner(kerastuner.tuners.BayesianOptimization):
    def run_trial(self, trial, *args, **kwargs):
    # You can add additional HyperParameters for preprocessing and custom training loops
    # via overriding `run_trial`
        kwargs['batch_size'] = trial.hyperparameters.Choice('batch_size', [512,1024,2048,4096])
        super(MyTuner, self).run_trial(trial, *args, **kwargs)

# Uses same arguments as the BayesianOptimization Tuner.
tuner = MyTuner(...)


# In[ ]:


class CVTuner(kerastuner.engine.tuner.Tuner):
  def run_trial(self, trial, x, y, batch_size=32, epochs=1):
    cv = model_selection.KFold(5)
    val_losses = []
    for train_indices, test_indices in cv.split(x):
      x_train, x_test = x[train_indices], x[test_indices]
      y_train, y_test = y[train_indices], y[test_indices]
      model = self.hypermodel.build(trial.hyperparameters)
      model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
      val_losses.append(model.evaluate(x_test, y_test))
    self.oracle.update_trial(trial.trial_id, {'val_loss': np.mean(val_losses)})
    self.save_model(trial.trial_id, model)
tuner = CVTuner(
  hypermodel=my_build_model,
  oracle=kerastuner.oracles.BayesianOptimization(
    objective='val_loss',
    max_trials=40))


# In[35]:


models.predict(X_test)


# In[60]:


nodes = [5,10,15]
batches=[1024,2048,4096]
epochs = [10,20,40]
param_options = {
                #'node': nodes,
                 'batch_size':batches,
               #  'epochs':epochs
                }     


# In[64]:


get_ipython().run_line_magic('load_ext', 'tensorboard')


# In[73]:


HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([5, 10]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.2, 0.3))

METRIC_ACCURACY = 'accuracy'

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
  hp.hparams_config(
    hparams=[HP_NUM_UNITS, HP_DROPOUT],
    metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
  )


# In[71]:


def train_test_model(hparams):
    MLP = Sequential([Dense(hparams[HP_NUM_UNITS],
                           activation='relu',
                           input_dim=X_train.shape[1],
                 #           kernel_initializer= kein 
                           ),
                      Dropout(hparams[HP_DROPOUT]),
                      Dense(5, activation='softmax')])
    MLP.compile(
      optimizer='adam',
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy'],
    )

    MLP.fit(X_res, y_res, epochs=1) # Run with 1 epoch to speed things up for demo purposes
    _, accuracy = MLP.evaluate(X_test, y_test)
    return accuracy


# In[68]:


def run(run_dir, hparams):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        accuracy = train_test_model(hparams)
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)


# In[74]:


session_num = 0

for num_units in HP_NUM_UNITS.domain.values:
  for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
      hparams = {
          HP_NUM_UNITS: num_units,
          HP_DROPOUT: dropout_rate,
      }
      run_name = "run-%d" % session_num
      print('--- Starting trial: %s' % run_name)
      print({h.name: hparams[h] for h in hparams})
      run('logs/hparam_tuning/' + run_name, hparams)
      session_num += 1


# In[82]:


get_ipython().run_line_magic('tensorboard', '--logdir logs/hparam_tuning --host localhost --port=8081')


# In[79]:


get_ipython().run_line_magic('tensorboard', '--logdir logs/hparam_demo')


# In[63]:


model = KerasClassifier(build_fn = create_mlp,verbose = 2)
grid = GridSearchCV(estimator=model, 
                    param_grid=param_options,
                    scoring=gmean_scorer,
                    cv=3,
                    refit='GM')

start_time=time.time()

grid_result = grid.fit(X_res, y_res,
                       callbacks=[early_stops('accuracy')],
                       validation_data = (X_val,y_val),
                       )
end_time=time.time()

print((end_time-start_time)/60)


# In[51]:


gs_result = pd.DataFrame.from_dict(grid_result.cv_results_)
gs_result


# In[52]:


gs_result.to_excel('GS_resample_3metrics.xlsx')


# In[53]:


for i in range(3):
    model = create_mlp(node=20)
    start = time.time()
    monitor = model.fit(X_res, y_res,
                        callbacks=[early_stops('accuracy')],
                        validation_data=(X_val, y_val.values),
                        batch_size= 1024,
                        verbose=VERBOSE, epochs=20,
                       )
    end = time.time()
    # use the model to make predictions with the test data
    Y_pred = model.predict(X_test)
    rsts = rsts.append(get_accs(y_test.values,Y_pred,end-start,'MLP-HS-HP#1.'+str(i+1)))
rsts


# In[68]:


initial_bias = np.log([np.bincount(y)[-1]/np.bincount(y)[0]])
initial_bias


# In[ ]:





# In[ ]:





# In[35]:


tf.keras.initializers.VarianceScaling(
    scale=1.0, mode="fan_in", distribution="truncated_normal", seed=None
)


# In[36]:


from tensorflow.keras.initializers import RandomNormal,VarianceScaling


# In[60]:


he_ini = tf.keras.initializers.VarianceScaling(scale=2, mode='fan_in', distribution='normal',seed=35)


# # Ordinal multiclass
# 

# In[74]:


early_stop = {'accuracy':  tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
              'cohen_kappa': CohenKappa(num_classes=5,sparse_labels=True),
              'f1_score': F1Score(num_classes=5,average="micro",threshold=0.5),
               }


# In[75]:


def create_mlp(metric):
    MLP = Sequential([Dense(10,
                           activation='relu',
                           input_dim=X_train.shape[1],
                            kernel_initializer=he_ini,
                           ),
                        BatchNormalization(),
                      Dropout(0.5),
                      Dense(5, activation='softmax')])
    MLP.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                #loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                metrics=[metric]
               )
    return MLP


# In[76]:


for name, metric in early_stop.items():
    print(name)
    model = create_mlp(metric)
    start = time.time()
    monitor = model.fit(X_train, y_train.values,
                    callbacks=[early_stops(name)],
                    class_weight = cls_wgt,
                    validation_data=(X_val, y_val.values),
                    batch_size=BATCH_SIZE,
                verbose=1, epochs=EPOCH)
    end = time.time()
    rsts = rsts.append(evaluation(model, monitor, end - start,  'MLP '+ name))
print(rsts.iloc[:,-4:])


# # One-hot encoded multiclass
# 

# In[77]:


early_stop = {'auc':       tf.keras.metrics.AUC(name='auc'),
    'accuracy':  tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
              'precision': tf.keras.metrics.Precision(name='precision'),
              'recall':    tf.keras.metrics.Recall(name='recall'),
              
              }


# In[80]:


def create_mlp(metric):
    MLP = Sequential([Dense(10,
                           activation='relu',
                           input_dim=X_train.shape[1],
                            kernel_initializer=he_ini,
                           ),
                        BatchNormalization(),
                      Dropout(0.5),
                      Dense(5, activation='softmax')])
    MLP.compile(optimizer='adam',
                #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                metrics=[metric]
               )
    return MLP


# In[31]:


#rsts = rsts.iloc[:6,:].copy()


# In[82]:


for name, metric in early_stop.items():
    print(name)
    model = create_mlp(metric)
    start = time.time()
    monitor = model.fit(X_train, pd.get_dummies(y_train).values,
                    callbacks=[early_stops(name)],
                    class_weight = cls_wgt,
                    validation_data=(X_val, pd.get_dummies(y_val).values),
                    batch_size=BATCH_SIZE,
                    verbose=0, epochs=EPOCH)
    end = time.time()
    rsts = rsts.append(evaluation(model, monitor, end - start,  'MLP 1H '+ name))
print(rsts.iloc[:,-4:])


# In[33]:


rsts.to_csv('VCA_Early_stopping.csv')

