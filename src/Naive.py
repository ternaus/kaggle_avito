from __future__ import division

__author__ = 'Vladimir Iglovikov'

import pandas as pd
import sqlite3
from pandas.io import sql
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import time

import numpy as np

conn = sqlite3.connect('../data/database.sqlite')

# query = """
# select Position, HistCTR, IsClick from trainSearchStream where ObjectType = 3 limit {N}, {N};
# """.format(N=3200000)

query = """
select Position, HistCTR, IsClick from trainSearchStream where ObjectType = 3;
""".format(N=10000000)


df = sql.read_sql(query, conn)
X = df[['Position', 'HistCTR']]
y = df.IsClick

ind = 1

if ind == 1:

  # Get test data
  query_test = """
  select TestID, Position, HistCTR from testSearchStream where ObjectType = 3
  """
  df_test = sql.read_sql(query_test, conn)
  X_test = df_test[['Position', 'HistCTR']]

  random_state = 42

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

