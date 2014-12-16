#!/usr/bin/env python
#-*-coding:utf-8-*-


import csv
from parser import parser
from math   import exp
file_name = "csvtest/20141201231852094000_Avg.csv"


def f(x):
  """
  if x < 0.001:
    return 0
  elif x > 0.999:
    return 1
  """
  return x


def g(x):
  return 1/ (1 + exp(-x))


spam_reader = parser(file_name)
result = []
count = 0
for row in spam_reader:
  count += 1
  if count == 1:
    continue
  row[1] = f(float(row[1]))
  result.append(row)



b = open('lol.csv', 'w')
a = csv.writer(b)
a.writerows(result)
b.close()
