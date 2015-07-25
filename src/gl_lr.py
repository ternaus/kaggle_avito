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
from sklearn.preprocessing import StandardScaler
import time
import pandas as pd

min_date = gl.SArray(['2015-05-10']).str_to_datetime()[0]

jan1 = gl.SArray(['2015-01-01']).str_to_datetime()[0]


print 'reading train'
train = gl.SFrame(os.path.join('..', 'data', 'train_ads_search'))[:10**5]

print
print 'train shape'
print train.shape

print 'filling missing values in Price'
a = train['Price'].mean()
train = train.fillna('Price', a)

features = ['Position',
          'HistCTR',
          'Price',
          'CategoryID',
          # 'RegionID',
          'RegionID.1',
          # 'CityID',
          'CategoryID.1',
          # 'AdID',
          # 'LocationID'
            # 'month',
            'day',
            'weekday',
            'days',
            'hour',
            'IsUserLoggedOn',
            'LocationID.1',            
          ]

# train = train[:10**8]

train = train[train['SearchDate'] > min_date]

# train_cut['month'] = train_cut['SearchDate'].apply(lambda x: x.month)
train['day'] = train['SearchDate'].apply(lambda x: str(x.day))
train['weekday'] = train['SearchDate'].apply(lambda x: str(x.weekday()))
train['days'] = train['SearchDate'].apply(lambda x: (x - jan1).days)
train['hour'] = train['SearchDate'].apply(lambda x: str(x.hour))
train['CategoryID'] = train['CategoryID'].apply(lambda x: str(x))
train['CategoryID.1'] = train['CategoryID.1'].apply(lambda x: str(x))
# train['LocationID.1'] = train['LocationID.1'].apply(lambda x: str(x))
train['RegionID'] = train['RegionID'].apply(lambda x: str(x))
train['RegionID.1'] = train['RegionID.1'].apply(lambda x: str(x))
train['CityID'] = train['CityID'].apply(lambda x: str(x))
train['CityID.1'] = train['CityID.1'].apply(lambda x: str(x))

print train.shape

ind = 2
if ind == 1:
	y = list(train['IsClick'])

	X = train[features].to_dataframe()

	print X.head()
	print X.info()

	print X.shape, len(y)

	scaler = StandardScaler()
	X = scaler.fit_transform(X)

	random_state = 42

	# Learn
	clf = LogisticRegression(random_state=random_state)

	skf = cross_validation.StratifiedKFold(y, n_folds=5, shuffle=True, random_state=random_state)
	scores = cross_validation.cross_val_score(clf, X, y, cv=skf, scoring='log_loss', n_jobs=-1)
	print -np.mean(scores), np.std(scores)
elif ind == 2:
  print 'fitting'
  model = gl.logistic_classifier.create(train,
                                        target='IsClick', features=features)
  print model.summary()                                       


  print 'reading test'
  test = gl.SFrame(os.path.join('..', 'data', 'test_ads_search'))

  print 'adding data features'

  test['SearchDate'] = test['SearchDate'].str_to_datetime()

  test['day'] = test['SearchDate'].apply(lambda x: str(x.day))
  test['hour'] = test['SearchDate'].apply(lambda x: str(x.hour))
  test['weekday'] = test['SearchDate'].apply(lambda x: str(x.weekday()))
  test['days'] = test['SearchDate'].apply(lambda x: (x - jan1).days)
  test['CategoryID'] = test['CategoryID'].apply(lambda x: str(x))
  test['CategoryID.1'] = test['CategoryID.1'].apply(lambda x: str(x))
  # test['LocationID.1'] = test['LocationID.1'].apply(lambda x: str(x))
  test['RegionID'] = test['RegionID'].apply(lambda x: str(x))
  test['RegionID.1'] = test['RegionID.1'].apply(lambda x: str(x))

  test['CityID'] = test['CityID'].apply(lambda x: str(x))
  test['CityID.1'] = test['CityID.1'].apply(lambda x: str(x))


  test = test.fillna('Price', a)
  print test.shape
  print 'predicting'
  prediction = model.predict(test, output_type='probability')
  result = pd.DataFrame()
  result['IsClick'] = prediction
  result['ID'] = test['ID']
  result.to_csv('predictions/LR_{timestamp}.csv'.format(timestamp=time.time()), index=False)

