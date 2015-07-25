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
print 'convert date to datetime format'
search['SearchDate'] = search['SearchDate'].str_to_datetime()
print 'cropping search'
min_date = gl.SArray(['2015-05-01']).str_to_datetime()[0]
search = search[search['SearchDate'] >= min_date]
print 'search shape'
print search.shape

print 'reading category'
category = gl.SFrame(os.path.join('..', 'data', 'category_1'))
print
print 'category shape'
print category.shape

print 'reading location'
location = gl.SFrame(os.path.join('..', 'data', 'location_1'))
print
print 'location shape'
print location.shape

print 'reading UserInfo'
user = gl.SFrame(os.path.join('..', 'data', 'user_1'))

print
print 'user shape'
print user.shape


print 'merging train ads and category'
train = train.join(category, how='left',
                   # on={'CategoryID', 'CategoryID'}
                   )
print 'merging test ads and category'
test = test.join(category, how='left',
                 # on={'CategoryID', 'CategoryID'}
                 )

print 'merging train ads category and location'
train = train.join(location, how='left',
                   # on={'LocationID', 'LocationID'}
                   )

print 'merging test ads category and location'
test = test.join(location, how='left',
                 # on={'LocationID', 'LocationID'}
                 )

print 'merging search and location'
search = search.join(location, how='left',
                     # on={'LocationID', 'LocationID'}
                     )

print 'merging search and category'
search = search.join(category, how='left',
                     # on={'CategoryID', 'CategoryID'}
                     )

print 'merging train and search'
training = train.join(search, how='left', on={'SearchID': 'SearchID',
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


