from __future__ import division

__author__ = 'Vladimir Iglovikov'

'''
I will try to use logistic regression from Graphlab to predict
'''

import graphlab as gl
import os

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
