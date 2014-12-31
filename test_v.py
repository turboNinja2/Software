#!/usr/bin/env python
#-*-coding:utf-8-*-


import csv
from parser import parser
from math   import *
from tools.misc import logloss
from density import Density
file_name1 = "csvtest/20141201231852094000_Avg.csv"
file_name2 = "csvtest/validation_set.csv"
#file_name1 = "lol.csv"

def f(x):
  #return max(min(0.95,x),0.05)
  if x < 0.005:
    return 0
  elif x > 0.99:
    return 1
  return x


d = Density()
d.run()
f = d.adjust
print "model build"


spam_reader1 = parser(file_name1)
spam_reader2 = parser(file_name2)
score = 0
count = 0
bad = 0
good = 0
for row1 in spam_reader1:
  row2 = spam_reader2.next()
  count += 1
  if count == 1:
    continue
  score += logloss(f(float(row1[1])),float(row2[1]))

print score/count
print "good : %s, bad : %s" % (good, bad)


b = open('lol.csv', 'w')
a = csv.writer(b)
a.writerows(result)
b.close()
