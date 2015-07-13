from __future__ import division

__author__ = 'Vladimir Iglovikov'

'''
Right now db is 38379Mb,

let's slightly clean it
'''
import sqlite3

conn = sqlite3.connect('../data/database.sqlite')
c = conn.cursor()
query = "DELETE FROM trainSearchStream WHERE ObjectType <> 3;"
mydata = c.execute(query)

c.close()
conn.close()