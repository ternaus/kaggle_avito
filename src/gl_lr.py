from __future__ import division

__author__ = 'Vladimir Iglovikov'

'''
I will try to use logistic regression from Graphlab to predict
'''

import graphlab as gl
import os
import scipy as sp
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

def llfun(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1 - epsilon, pred)
    ll = sum(act * sp.log(pred) + sp.subtract(1, act) * sp.log(sp.subtract(1, pred)))
    ll = ll * -1.0/len(act)
    return ll

print 'reading train'
train = gl.SFrame(os.path.join('..', 'data', 'train_ads_search'))

print
print 'train shape'
print train.shape

print 'filling missing values'
a = train['Price'].mean()
train = train.fillna('Price', a)

features = ['Position',
          'HistCTR',
          'Price',
          # 'CategoryID',
          # 'AdID',
          # 'LocationID'
          ]

X = train[:10**6][features].to_dataframe()
y = train[:10**6]['IsClick']

random_state = 42

# Learn
clf = LogisticRegression(random_state=random_state)

skf = cross_validation.StratifiedKFold(y, n_folds=5, shuffle=True, random_state=random_state)
scores = cross_validation.cross_val_score(clf, X, y, cv=skf, scoring='log_loss', n_jobs=-1)
print -np.mean(scores), np.std(scores)

# print 'splitting set'
# sf_train, sf_test = train.random_split(0.5, seed=42)
#
# features=['Position',
#           'HistCTR',
#           'Price',
#           # 'CategoryID',
#           # 'AdID',
#           # 'LocationID'
#           ]
#
# model = gl.logistic_classifier.create(sf_train,
#                                       target='IsClick',
#                                       features=features,
#                                       validation_set=sf_test)
#
# print 'predicting'
# prediction = model.predict(sf_test, output_type='probablility')
#
# print prediction[:30]
# print sf_test['IsClick'][:30]
# print 'calculating score'
# score = llfun(sf_test['IsClick'], prediction)
# print score