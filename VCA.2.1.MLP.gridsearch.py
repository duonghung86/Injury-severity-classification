# Basic packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time

# Preprocessing
from sklearn.preprocessing import StandardScaler # Standardization
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Machine learning algos
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
# Metrics
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import auc
# Tensorflow
import tensorflow as tf
from tensorflow import feature_column  # for data wrangling
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers
from tensorflow.keras.initializers import Zeros, RandomNormal, RandomUniform
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.losses import SparseCategoricalCrossentropy, KLDivergence, Poisson
from tensorflow.keras.optimizers import Adam
# Check tensorflow version
print('tensorflow version: ', tf.__version__)

# %% Import dataset
df = pd.read_csv('Prepared Texas 2019.csv')
print(df.shape)
df.head()
# create a back up
backup_df = df.copy()

# Let's just use a small part of the total dataset
df, _ = train_test_split(df, test_size=0.99, stratify=df['Prsn_Injry_Sev'])
print(df.shape)

y = df['Prsn_Injry_Sev']
print('All target values:', y.value_counts())
X = df.drop(columns=['Prsn_Injry_Sev'])

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

# %% Function to compare the prediction and true labels
def get_accs(label, prediction, show=True):
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

# get_accs(np.random.randint(5, size=100), np.random.randint(5, size=100))

# %% Traditional ML

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier

cls_wgt = 'balanced'
LR = LogisticRegression(solver='lbfgs', class_weight=cls_wgt)
DT = DecisionTreeClassifier(class_weight=cls_wgt)
RF = RandomForestClassifier(max_depth=4, class_weight=cls_wgt)
GNB = GaussianNB()
SGD = SGDClassifier(loss="hinge", penalty="l2", max_iter=5, class_weight=cls_wgt)

clfs = [LR, DT, RF]
clf_names = ['LR', 'DT', 'RF']
rsts = pd.DataFrame()
for model, name in zip(clfs, clf_names):
    start = time.time()
    print(name)
    model.fit(X_train, y_train.values)
    # use the model to make predictions with the test data
    y_pred = model.predict(X_test)
    end = time.time()
    # get the evaluation metrics
    result = get_accs(y_test.values, y_pred, True)
    result['Training Time'] = np.round(end - start, 3)
    result.index = [name]
    rsts = rsts.append(result)
rsts

# %% Model 0 with weight
# Add weights
weights = len(y) / (5 * np.bincount(y))
cls_wgt = dict(zip(np.arange(5), weights))
cls_wgt


# %% function to display the evolution of training process:
def show_evolution(moni):
    hist = pd.DataFrame(moni.history)
    # names = hist.columns
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 2), dpi=150)
    hist[['loss', 'val_loss']].plot(ax=axes[0])
    hist[['accuracy', 'val_accuracy']].plot(ax=axes[1])
    plt.show()


# %% create model
def create_mlp():
    MLP = tf.keras.Sequential([
        keras.layers.Dense(10,
                           activation='relu',
                           input_dim=X_train.shape[1]
                           ),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(5, activation='softmax')])
    MLP.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    return MLP


# %% Produce an evaluation on the MLP model
def evaluation(model, monitor, time, name):
    # use the model to make predictions with the test data
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    # Show evolution of the training process
    show_evolution(monitor)
    # get the evaluation metrics
    result = get_accs(y_test.values, y_pred, False)
    result['Training Time'] = np.round(time, 3)
    result.index = [name]
    return result
# %% Setup initial parameters
# Early stopping
es = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                      verbose=1,
                                      patience=10,
                                      mode='max',
                                      restore_best_weights=True)

# %% baseline model
model = create_mlp()
start = time.time()
monitor = model.fit(X_train, y_train.values,
                        callbacks=[es],
                        validation_data=(X_val, y_val),
                        verbose=1, epochs=50)
end = time.time()

result = evaluation(model, monitor, end-start, 'MLP baseline')
rsts = rsts.append(result)

# %% Model with class weights

model = create_mlp()
start = time.time()
monitor = model.fit(X_train, y_train.values,
                        callbacks=[es],
                        validation_data=(X_val, y_val),
                        verbose=1, epochs=50)
end = time.time()

result = evaluation(model, monitor, end-start, 'MLP with weights')
rsts = rsts.append(result)

print(rsts)

# %% Parameter tunings

# %%% initial weight

bias_zeros = initializers.Zeros()
bias_rn = initializers.RandomNormal(mean=0.0, stddev=0.005, seed=None)
bias_ru = initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
bias_tn = initializers.TruncatedNormal(mean=0., stddev=1.)
bias_ones = initializers.Ones()
biases = [bias_zeros, bias_rn, bias_ru, bias_tn,
          bias_ones]
bias_name = ['bias_zeros', 'bias_rn', 'bias_ru', 'bias_tn',
             'bias_ones']

def create_mlp(ini_bias):
    MLP = tf.keras.Sequential([
        keras.layers.Dense(10,
                           activation='relu',
                           input_dim=X_train.shape[1],
                           bias_initializer=ini_bias
                           ),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(5, activation='softmax')])
    MLP.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    return MLP

for bias, name in zip(biases, bias_name):
    print(name)
    try:
        model = create_mlp(bias)
    except Exception as e:
        print(e)
        continue
    start = time.time()
    monitor = model.fit(X_train, y_train, callbacks=[es], validation_data=(X_val, y_val),
                        class_weight=cls_wgt,
                        verbose=1, epochs=50)
    end = time.time()
    rsts = rsts.append(evaluation(model, monitor, end - start, name))

print(rsts)

# %% regularizers
reg1 = l1(0.0001)
reg2 = l2(0.0001)
l12 = l1_l2(0.0001, 0.0001)
regus = [None, reg1, reg2, l12]
regu_names = ['None', "l1", 'l2', 'l12']

ini_bias = initializers.RandomUniform(minval=-1e-05, maxval=1e-05, seed=None)
def create_mlp(r1, r2):
    MLP = tf.keras.Sequential([
        keras.layers.Dense(10,
                           activation='relu',
                           input_dim=X_train.shape[1],
                           bias_initializer=ini_bias,
                           activity_regularizer=r1
                           ),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(5, activation='softmax',
                           activity_regularizer=r2)])
    MLP.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    return MLP

for regu1, name1 in zip(regus, regu_names):
    for regu2, name2 in zip(regus, regu_names):
        name = 'Re ' + name1 + ' and ' + name2
        print(name)
        model = create_mlp(regu1, regu2)
        start = time.time()
        monitor = model.fit(X_train, y_train, callbacks=[es], validation_data=(X_val, y_val),
                            class_weight=cls_wgt,
                            verbose=0, epochs=50)
        end = time.time()
        rsts = rsts.append(evaluation(model, monitor, end - start, name))

# %% activation
activations = ['relu', 'sigmoid', 'tanh', 'selu', 'elu', 'exponential']

def create_mlp(acti):
    MLP = tf.keras.Sequential([
        keras.layers.Dense(10,
                           activation=acti,  # tuning parameter
                           input_dim=X_train.shape[1],
                           bias_initializer=ini_bias,
                           activity_regularizer=l1(0.0001)
                           ),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(5, activation='softmax',
                           activity_regularizer=l1_l2(0.0001, 0.0001))])
    MLP.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    return MLP

for acti in activations:
    print(acti)
    model = create_mlp(acti)
    start = time.time()
    monitor = model.fit(X_train, y_train, callbacks=[es], validation_data=(X_val, y_val),
                        class_weight=cls_wgt,
                        verbose=0, epochs=50)
    end = time.time()
    rsts = rsts.append(evaluation(model, monitor, end - start, acti))


# %% kernel_initializer

def create_mlp(ker_ini):
    MLP = tf.keras.Sequential([
        keras.layers.Dense(10,
                           activation='relu',  # tuning parameter
                           input_dim=X_train.shape[1],
                           bias_initializer=ini_bias,
                           activity_regularizer=l1(1e-4),
                           kernel_initializer=ker_ini
                           ),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(5, activation='softmax',
                           activity_regularizer=l1_l2(1e-4, 1e-4))])
    MLP.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    return MLP

for bias, name in zip(biases, bias_name):
    print(name)
    try:
        model = create_mlp(bias)
    except Exception as e:
        print(e)
        continue
    start = time.time()
    monitor = model.fit(X_train, y_train, callbacks=[es], validation_data=(X_val, y_val),
                        class_weight=cls_wgt,
                        verbose=1, epochs=5)
    end = time.time()
    rsts = rsts.append(evaluation(model, monitor, end - start, name))


# %% loss function 1

scc1 = SparseCategoricalCrossentropy(from_logits=True)
scc2 = SparseCategoricalCrossentropy()
kld = KLDivergence()
poi = Poisson()
losses = [scc1, scc2, kld, poi]
loss_names = ['scc1', 'scc2', 'kld', 'poi']

def create_mlp(los_fun):
    MLP = tf.keras.Sequential([
        keras.layers.Dense(10,
                           activation='relu',
                           input_dim=X_train.shape[1],
                           bias_initializer=RandomUniform(minval=-1e-05, maxval=1e-05, seed=None),
                           activity_regularizer=l1(1e-4),
                           ),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(5, activation='softmax',
                           activity_regularizer=l1_l2(1e-4, 1e-4))])
    MLP.compile(optimizer='adam',
                loss=los_fun,  # tuning parameter
                metrics=['accuracy'])
    return MLP

for func, name in zip(losses, loss_names):
    print(name)
    try:
        model = create_mlp(func)
    except Exception as e:
        print(e)
        continue
    start = time.time()
    monitor = model.fit(X_train, y_train, callbacks=[es], validation_data=(X_val, y_val),
                        class_weight=cls_wgt,
                        verbose=1, epochs=5)
    end = time.time()
    rsts = rsts.append(evaluation(model, monitor, end - start, name))

# %% Optimizer

# Learning rate
adams = [Adam(learning_rate=1 / 10 ** x) for x in range(2, 6)]
names = ['lr #{}'.format(1 / 10 ** x) for x in range(2, 6)]
adams, names

def create_mlp(opti):
    MLP = tf.keras.Sequential([
        keras.layers.Dense(10,
                           activation='relu',
                           input_dim=X_train.shape[1],
                           bias_initializer=RandomUniform(minval=-1e-05, maxval=1e-05, seed=None),
                           activity_regularizer=l1(1e-4),
                           ),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(5,
                           activation='softmax',
                           activity_regularizer=l1_l2(1e-4, 1e-4))])
    MLP.compile(optimizer=opti,  # tuning paramete
                loss=SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    return MLP

for optimizer, name in zip(adams, names):
    print(name)
    try:
        model = create_mlp(optimizer)
    except Exception as e:
        print(e)
        continue
    start = time.time()
    monitor = model.fit(X_train, y_train.values, callbacks=[es], validation_data=(X_val, y_val),
                        class_weight=cls_wgt,
#                        batch_size=32,
                        verbose=1, epochs=5)
    end = time.time()
    rsts = rsts.append(evaluation(model, monitor, end - start, name))

# %% Epsilon
rang = range(6, 10)
adams = [Adam(learning_rate=0.001, epsilon=1 / 10 ** x) for x in rang]
names = ['epsilon #{}'.format(1 / 10 ** x) for x in rang]
adams, names

for optimizer, name in zip(adams, names):
    print(name)
    try:
        model = create_mlp(optimizer)
    except Exception as e:
        print(e)
        continue
    start = time.time()
    monitor = model.fit(X_train, y_train.values, callbacks=[es], validation_data=(X_val, y_val),
                        class_weight=cls_wgt,
  #                      batch_size=32,
                        verbose=1, epochs=5)
    end = time.time()
    rsts = rsts.append(evaluation(model, monitor, end - start, name))

# %% beta_1=0.9
rang = [0.93 + 0.01 * x for x in range(5)]
adams = [Adam(beta_1=x) for x in rang]
names = ['beta_1 #{}'.format(x) for x in rang]
adams, names

for optimizer, name in zip(adams, names):
    print(name)
    try:
        model = create_mlp(optimizer)
    except Exception as e:
        print(e)
        continue
    start = time.time()
    monitor = model.fit(X_train, y_train.values, callbacks=[es], validation_data=(X_val, y_val),
                        class_weight=cls_wgt,
  #                      batch_size=32,
                        verbose=1, epochs=5)
    end = time.time()
    rsts = rsts.append(evaluation(model, monitor, end - start, name))


# %% Node number

def create_mlp(node):
    MLP = tf.keras.Sequential([
        keras.layers.Dense(node,  # tuning paramete
                           activation='relu',
                           input_dim=X_train.shape[1],
                           bias_initializer=RandomUniform(minval=-1e-05, maxval=1e-05, seed=None),
                           activity_regularizer=l1(1e-4),
                           ),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(5,
                           activation='softmax',
                           activity_regularizer=l1_l2(1e-4, 1e-4))])
    MLP.compile(optimizer=Adam(),
                loss=SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    return MLP

for i in range(5, 21, 5):
    print(i)
    try:
        model = create_mlp(i)
    except Exception as e:
        print(e)
        continue
    start = time.time()
    monitor = model.fit(X_train, y_train.values, callbacks=[es], validation_data=(X_val, y_val),
                        class_weight=cls_wgt,
#                        batch_size=32,
                        verbose=1, epochs=5)
    end = time.time()
    rsts = rsts.append(evaluation(model, monitor, end - start, str(i)))

print(rsts)

# Export the outcome
rsts.to_csv('Evaluation {}.csv'.format(time.ctime()),index=False)