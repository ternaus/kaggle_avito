from __future__ import division

__author__ = 'Vladimir Iglovikov'

'''
I will try to use logistic regression from Graphlab to predict
'''

import graphlab as gl
import os
import scipy as sp
import math

def llfun(act, pred):
    N = len(act)
    result = 0
    for i in range(N):
      result += act[i] * math.log(pred[i]) + (1 - act[i]) * math.log(1 - pred[i])
    #
    # epsilon = 1e-15
    # pred = sp.maximum(epsilon, pred)
    # pred = sp.minimum(1 - epsilon, pred)
    # ll = sum(act * sp.log(pred) + sp.subtract(1,act) * sp.log(sp.subtract(1, pred)))
    # ll = ll * -1.0/len(act)
    return -result / N

print 'reading train'
train = gl.SFrame(os.path.join('..', 'data', 'train_ads_search'))

print
print 'train shape'
print train.shape

print 'filling missing values'
a = train['Price'].mean()
train = train.fillna('Price', a)

print 'splitting set'
sf_train, sf_test = train.random_split(0.5, seed=42)

features=['Position',
          'HistCTR',
          'Price',
          # 'CategoryID',
          # 'AdID',
          # 'LocationID'
          ]

model = gl.logistic_classifier.create(sf_train,
                                      target='IsClick',
                                      features=features,
                                      validation_set=sf_test)

score = llfun(sf_test['IsClick'], model.predict(sf_test, output_type='probablility'))
print score