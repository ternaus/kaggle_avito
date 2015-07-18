from __future__ import division

__author__ = 'Vladimir Iglovikov'

'''
It takes too long to merge datasets every time. I need to merge it before doing regression
'''

import graphlab as gl
import os

print 'reading train'
train = gl.SFrame(os.path.join('..', 'data', 'trainSearch_1'))

print
print 'train shape'
print train.shape

print 'reading test'
test = gl.SFrame(os.path.join('..', 'data', 'testSearch_1'))
print
print 'test shape'
print test.shape



print 'reading AdsInfo'
ads = gl.SFrame(os.path.join('..', 'data', 'ads_1'))
print
print 'ads shape'
print ads.shape

print 'merging train and ads'
training = train.join(ads, on='AdID')

print
print 'training shape'
print training.shape

print 'merging train and ads'
testing = test.join(ads, on='AdID')

print
print 'testing shape'
print testing.shape

print 'saving train'
training.save(os.path.join('..', 'data', 'train_ads'))

print 'saving test'
testing.save(os.path.join('..', 'data', 'test_ads'))


