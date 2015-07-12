__author__ = 'Vladimir Iglovikov'

"""

This is sample script that gives 0.05045 at the LB, while maximum score so far is
 0.04077

Logistic Regression

https://www.kaggle.com/olivermeyfarth/avito-context-ad-clicks/logistic-regression-on-histctr/run/19490
"""

import sqlite3
import datetime
import zipfile
import pandas as pd
import numpy as np
from pandas.io import sql
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer

conn = sqlite3.connect('../data/database.sqlite')

# Get train data
query = """
select Position, HistCTR, IsClick from trainSearchStream where ObjectType = 3 limit 10000, 10000;
"""
df = sql.read_sql(query, conn)
X = df[['Position', 'HistCTR']]
y = df.IsClick

# Get test data
query_test = """
select TestID, Position, HistCTR from testSearchStream where ObjectType = 3
"""
df_test = sql.read_sql(query_test, conn)
X_test = df_test[['Position', 'HistCTR']]

# Learn
model = LogisticRegression()
model.fit(X, y)
pred = model.predict_proba(X_test)

# Output to csv
filename = 'submission.csv'
pd.DataFrame({'ID': df_test.TestId, 'IsClick': pred[:, 1]}).to_csv(filename, index=False)

# Zip
# with zipfile.ZipFile(filename + '.zip', 'w', zipfile.ZIP_DEFLATED) as z:
#     z.write(filename)

