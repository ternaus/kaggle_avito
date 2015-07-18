from __future__ import division

__author__ = 'Vladimir Iglovikov'

'''
Dataset is big for my laptop => I will try slightly to clean and change format of the data
'''

import graphlab as gl
import os
import sys

data_path = os.path.join('..', 'data')


print 'working with train'
train = gl.SFrame(os.path.join(data_path, 'trainSearchStream.tsv'))
train_new = train[train['ObjectType'] == 3]
del train_new['ObjectType']
train_new.save(os.path.join(data_path, 'trainSearch_1'))

print 'working with test'
test = gl.SFrame(os.path.join(data_path, 'testSearchStream.tsv'))
test_new = test[test['ObjectType'] == 3]
del test_new['ObjectType']
test_new.save(os.path.join(data_path, 'testSearch_1'))

print 'cleaning VisitsStream'
visits = gl.SFrame(os.path.join(data_path, 'VisitsStream.tsv'))
visits.save(os.path.join(data_path, 'visits_1'))

print 'cleaning SearchInfo'
search = gl.SFrame(os.path.join(data_path, 'SearchInfo.tsv'))

search['SearchDate'] = search['SearchDate'].str_to_datetime()

search.save(os.path.join(data_path, 'search_1'))

print 'clean userinfo'
user = gl.SFrame(os.path.join(data_path, 'UserInfo.tsv'))
user.save(os.path.join(data_path, 'user_1'))

print 'clean PhoneRequestsStream'
phone = gl.SFrame(os.path.join(data_path, 'PhoneRequestsStream.tsv'))
phone.save(os.path.join(data_path, 'phone_1'))

print 'clean Category'
user = gl.SFrame(os.path.join(data_path, 'Category.tsv'))
user.save(os.path.join(data_path, 'category_1'))

print 'clean Location'
user = gl.SFrame(os.path.join(data_path, 'Location.tsv'))
user.save(os.path.join(data_path, 'location_1'))

print 'clean AdsInfo'
user = gl.SFrame(os.path.join(data_path, 'AdsInfo.tsv'))
user.save(os.path.join(data_path, 'ads_1'))






