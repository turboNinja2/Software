#!/usr/bin/env python
#-*-coding:utf-8-*-


import csv
from parser import parser
from math   import *
from tools.misc import logloss
file_name1 = "csvtest/20141201231852094000_Avg.csv"
file_name2 = "csvtest/validation_set.csv"


def foo(index, vector,k=100):
  s = 0
  limit = False
  for i in xrange(k):
    try:
      s += vector[index - 1 + i][1]
    except:
      s = s/float(i)
      limit = True
      continue
  if not limit:
    s = s/float(k)
  return s

X= []

spam_reader1 = parser(file_name1)
spam_reader2 = parser(file_name2)
count = 0
for row1 in spam_reader1:
  row2 = spam_reader2.next()
  count += 1
  if count == 1:
    continue
  X.append((float(row1[1]),int(row2[1])))

X.sort(key=lambda x : x[0])
density = 0
result = [(0.0,0.0)]
for c,x in enumerate(X):
  result.append((x[0],foo(c,X,10)))
result.append((1.0,1.0))

b = open('lol.csv', 'w')
a = csv.writer(b)
a.writerows(result)
b.close()
