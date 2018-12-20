import numpy as np
import pandas as pd
from datetime import datetime
import itertools
from itertools import chain
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
pd.options.mode.chained_assignment = None
import gc

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
pd.set_option('max_columns', 40)

import os
print(os.listdir("../input"))


from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()

(market, news) = env.get_training_data()

# market = market.loc[market['time'] >= '2010-01-01 00:00:00+0000'].reset_index(drop = True)

### prepare market data
market['close/open'] = market['close'] / market['open']
market['assetOpenMedian'] = market.groupby('assetCode')['open'].transform('median')
market['assetCloseMedian'] = market.groupby('assetCode')['close'].transform('median')
    
for i, row in market.loc[market['close/open'] >= 1.5].iterrows():
    if np.abs(row['assetOpenMedian'] - row['open']) > np.abs(row['assetCloseMedian'] - row['close']):
        market.iloc[i,5] = row['assetOpenMedian']
    else:
        market.iloc[i,4] = row['assetCloseMedian']
        
for i, row in market.loc[market['close/open'] <= 0.5].iterrows():
    if np.abs(row['assetOpenMedian'] - row['open']) > np.abs(row['assetCloseMedian'] - row['close']):
        market.iloc[i,5] = row['assetOpenMedian']
    else:
        market.iloc[i,4] = row['assetCloseMedian']
        

fillna_cols = ['returnsClosePrevMktres1','returnsOpenPrevMktres1','returnsClosePrevMktres10','returnsOpenPrevMktres10']

market = market.sort_values(by = ['assetCode', 'time'], ascending = [True, True])
for i in market[fillna_cols]:
    market[i] = market[i].fillna(method = 'ffill')
    

market_train = market.drop(['assetOpenMedian','assetCloseMedian'], axis = 1)

num_cols = [c for c in market_train.columns if c not in['assetName','assetCode','time','universe']]
feat_cols = [c for c in market_train.columns if c not in['assetName','assetCode','time','universe','returnsOpenNextMktres10']]


scaler = StandardScaler()
market_train[num_cols] = scaler.fit_transform(market_train[num_cols])

train_indices, val_indices = train_test_split(market_train.index.values,test_size=0.05, random_state=23)

def get_input(market_df, indices):
    X = market_df.loc[indices, feat_cols].values
    y = (market_df.loc[indices,'returnsOpenNextMktres10'] >= 0).values
    r = market_df.loc[indices,'returnsOpenNextMktres10'].values
    u = market_df.loc[indices, 'universe']
    d = market_df.loc[indices, 'time'].dt.date
    return X,y,r,u,d
    
X_train,y_train,r_train,u_train,d_train = get_input(market_train, train_indices)
X_valid,y_valid,r_valid,u_valid,d_valid = get_input(market_train, val_indices)


# NN
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout

model = Sequential()
model.add(Dense(128, input_dim=12, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer = 'adam',
              metrics=['accuracy'])

model.summary()

# X_train, X_test = model_selection.train_test_split(X.index.values, test_size = 0.05, random_state=23)

from keras.callbacks import EarlyStopping, ModelCheckpoint

check_point = ModelCheckpoint('model.hdf5',verbose=True, save_best_only=True)
early_stop = EarlyStopping(patience=5,verbose=True)

model.fit(X_train,y_train.astype(int),
          validation_data=(X_valid,y_valid.astype(int)),
          epochs=2,
          batch_size = 128,
          verbose=True,
          callbacks=[early_stop,check_point])

# score = model.evaluate(X_valid,y_valid.astype(int), batch_size=128)
# score

import matplotlib.pyplot as plt

model.load_weights('model.hdf5')
confidence_valid = model.predict(X_valid)[:,0]*2-1
print(accuracy_score(confidence_valid>0,y_valid))
plt.hist(confidence_valid, bins='auto')
plt.title("predicted confidence")


r_valid = r_valid.clip(-1,1)
x_t_i = confidence_valid * r_valid * u_valid
data = {'day' : d_valid, 'x_t_i' : x_t_i}
df = pd.DataFrame(data)
x_t = df.groupby('day').sum().values.flatten()
mean = np.mean(x_t)
std = np.std(x_t)
score_valid = mean / std
print(score_valid)


### submission

n_days = 0

for (market_obs_df, news_obs_df, predictions_template_df) in env.get_prediction_days():
    n_days += 1
    if n_days % 50 == 0:
        print (n_days, end = ' ')
    
    
    market_obs_df['close/open'] = market_obs_df['close'] / market_obs_df['open']
    
    scaler = StandardScaler()
    
    feat_cols = [c for c in market_obs_df.columns if c not in['assetName','assetCode','time','universe']]
    
    market_obs_df[feat_cols] = scaler.fit_transform(market_obs_df[feat_cols])
    
    X_SUB = market_obs_df[feat_cols]

    for i in X_SUB.columns:
        X_SUB[i] = X_SUB[i].fillna(method = 'ffill')

    for i in X_SUB.columns:
        X_SUB[i] = X_SUB[i].fillna(method = 'bfill')
    
    confidence = model.predict(X_SUB.values)[:,0]*2-1
    
    preds = pd.DataFrame({'assetCode':market_obs_df['assetCode'], 'confidence':confidence})
    predictions_template_df = predictions_template_df.merge(preds,how='left').drop('confidenceValue',axis=1).fillna(0).rename(columns={'confidence':'confidenceValue'})
    
    env.predict(predictions_template_df)

env.write_submission_file()
