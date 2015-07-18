from __future__ import division

__author__ = 'Vladimir Iglovikov'

'''
It takes too long to merge datasets every time. I need to merge it before doing regression
'''

import graphlab as gl
import os

print 'reading train'
train = gl.SFrame(os.path.join('..', 'data', 'train_ads'))

print
print 'train shape'
print train.shape
print train.column_names()

print 'reading test'
test = gl.SFrame(os.path.join('..', 'data', 'test_ads'))
print
print 'test shape'
print test.shape
print test.column_names()


print 'reading SearchInfo'
search = gl.SFrame(os.path.join('..', 'data', 'search_1'))
print
print 'search shape'
print search.shape
print search.column_names()

print 'shuffling train'

# x = gl.cross_validation.shuffle(train)[:10**7]

print 'merging train and search'
training = train[:10**7].join(search, on={'SearchID': 'SearchID',
                                  # 'LocationID': 'LocationID',
                                  # 'CategoryID': 'CategoryID'
                                  })

print
print 'training shape'
print training.shape
print training.column_names()

print 'merging test and ads'
testing = test.join(search, on={'SearchID': 'SearchID',
                                # 'LocationID': 'LocationID',
                                # 'CategoryID': 'CategoryID'
                                })

print
print 'testing shape'
print testing.shape
print testing.column_names()

print 'saving train'
training.save(os.path.join('..', 'data', 'train_ads_search'))

print 'saving test'
testing.save(os.path.join('..', 'data', 'test_ads_search'))


