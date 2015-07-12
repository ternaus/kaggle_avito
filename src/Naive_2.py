from __future__ import division

__author__ = 'Vladimir Iglovikov'

import pandas as pd
import sqlite3
from pandas.io import sql
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
import numpy as np

conn = sqlite3.connect('../data/database.sqlite')

# query = """
# select Position, HistCTR, IsClick from trainSearchStream where ObjectType = 3 limit {N}, {N};
# """.format(N=3200000)

query = """
select Position, HistCTR, IsClick, Price, Params
from (trainSearchStream join AdsInfo using(AdID))
where ObjectType = 3
limit {N};
""".format(N=10**5)


print 'reading query'

df = sql.read_sql(query, conn)

def clean_price(x):
  if len(x.strip()) == 0:
    return np.nan
  else:
    return float(x)

df['Price'] = df['Price'].apply(clean_price, 1)
df['Price'] = df['Price'].interpolate()

def get_str_list(x):
  try:
    a = ' '.join(map(str, eval(x).keys()))
  except:
    a = ' '
  return a

df['params_new'] = (df['Params']
                    .apply(lambda x: get_str_list(x), 1)
                    )


vectorizer = CountVectorizer(input=u'content',
                             encoding=u'utf-8',
#                              decode_error=u'strict',
#                              strip_accents=None,
#                              lowercase=True,
                             preprocessor=None,
                             tokenizer=None,
                             stop_words=None,
#                              token_pattern=u'(?u)\b\w\w+\b',
                              token_pattern=u'\d+',
                             ngram_range=(1, 1),
                             analyzer=u'word',
                             max_df=100,
                             min_df=0,
                             max_features=None,
                             vocabulary=None,
                             binary=True)

params_X = vectorizer.fit_transform(df['params_new'])


X = df[['Position', 'HistCTR', 'Price']]
y = df.IsClick

print
print X.shape

print params_X.shape

X = np.concatenate((X.values, params_X.toarray()), 1)

scaler = StandardScaler()

X = scaler.fit_transform(X)

print X.shape
ind = 2


random_state = 42
clf = LogisticRegression(random_state=random_state)

if ind == 1:
  # Get test data
  query_test = """
  select TestID, Position, HistCTR from testSearchStream where ObjectType = 3
  """
  df_test = sql.read_sql(query_test, conn)
  X_test = df_test[['Position', 'HistCTR']]

  # Learn
  clf = LogisticRegression(random_state=random_state)

  clf .fit(X, y)
  pred = clf.predict_proba(X_test)

  # Output to csv
  filename = 'predictions/LR_10_6{timestamp}.csv'.format(timestamp=time.time())
  pd.DataFrame({'ID': df_test.TestId, 'IsClick': pred[:, 1]}).to_csv(filename, index=False)

if ind == 2:
  skf = cross_validation.StratifiedKFold(y, n_folds=5, shuffle=True, random_state=random_state)
  scores = cross_validation.cross_val_score(clf, X, y, cv=skf, scoring='log_loss', n_jobs=2)
  print -np.mean(scores), np.std(scores)

