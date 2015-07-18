from __future__ import division

__author__ = 'Vladimir Iglovikov'

'''
I will try to use logistic regression from Graphlab to predict
'''

import graphlab as gl
import os

print 'reading train'
train = gl.SFrame(os.path.join('..', 'data', 'trainSearch_1'))

print
print 'train shape'
print train.shape



print 'reading AdsInfo'
ads = gl.SFrame(os.path.join('..', 'data', 'ads_1'))
print
print 'ads shape'
print ads.shape

print 'merging train and ads'

training = train.join(ads, on='AdId')

print
print 'training shape'
print training.shape

print 'splitting set'
sf_train, sf_test = training.random_split(0.5,
                                          seed=42)

features=['Position',
          'HistCTR',
          'Price',
          'CategoryID',
          'AdID',
          'LocationID']

model = gl.logistic_classifier.create(training,
                                      target='IsClick',
                                      features=features)