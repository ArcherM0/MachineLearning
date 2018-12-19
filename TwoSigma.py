import numpy as np
import pandas as pd
from datetime import datetime
import itertools
from itertools import chain
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import gc

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
pd.set_option('max_columns', 40)

import os
print(os.listdir("../input"))


from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()

(market, news) = env.get_training_data()


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


X = market_train[feat_cols]
Y = (market_train.returnsOpenNextMktres10 >= 0).astype('int8')

# NN
from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential([
    Dense(32, input_shape=(784,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])

X_train, X_test = model_selection.train_test_split(X.index.values, test_size = 0.05, random_state=23)

from keras.callbacks import EarlyStopping, ModelCheckpoint

check_point = ModelCheckpoint('model.hdf5',verbose=True, save_best_only=True)
early_stop = EarlyStopping(patience=5,verbose=True)
model.fit(X.loc[X_train],Y.loc[X_train],
          validation_data=(X.loc[X_test],Y.loc[X_test]),
          epochs=4,
          verbose=True,
          callbacks=[early_stop,check_point]) 

### submission

n_days = 0

for (market_obs_df, news_obs_df, predictions_template_df) in env.get_prediction_days():
    n_days += 1
    if n_days % 50 == 0:
        print (n_days, end = ' ')
        
    market_obs_df['close/open'] = market_obs_df['close']/market_obs_df['open']
    
    X_SUB = market_obs_df[feat_cols]
    
    for i in X_SUB.columns:
        X_SUB[i] = X_SUB[i].fillna(method = 'ffill')

    for i in X_SUB.columns:
        X_SUB[i] = X_SUB[i].fillna(method = 'bfill')
 
    news_obs_df = news_prep(news_obs_df)
    news_index = unstack_assetcodes(news_obs_df)
    news_merge = merge_news_index(news_obs_df, news_index)
    news_merge_value = group_news_value(news_merge)
    market_obs_df['close/open'] = market_obs_df['close']/market_obs_df['open']
    market_news_merge = merge_data(market_obs_df, news_merge_value)


    X_SUB = market_news_merge[feat_cols]
    
    for i in X_SUB.columns:
        X_SUB[i] = X_SUB[i].fillna(method = 'ffill')

    for i in X_SUB.columns:
        X_SUB[i] = X_SUB[i].fillna(method = 'bfill')
    
  
    lp = lgb.predict_proba(X_SUB.values)
    confidence = 2 * lp[:,1] - 1
    
    preds = pd.DataFrame({'assetCode':market_obs_df['assetCode'], 'confidence':confidence})
    predictions_template_df = predictions_template_df.merge(preds,how='left').drop('confidenceValue',axis=1).fillna(0).rename(columns={'confidence':'confidenceValue'})
    
    env.predict(predictions_template_df)

env.write_submission_file()

