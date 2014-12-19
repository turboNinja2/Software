#!/usr/bin/env python
#-*-coding:utf-8-*-


import csv
from parser import parser
from math   import *
from tools.misc import logloss
file_name1 = "csvtest/20141201231852094000_Avg.csv"
file_name2 = "csvtest/validation_set.csv"


class Density():

  ####################################################################################
  ## INIT FUNCTIONS

  def __init__(self):
    self.X = []
    self._file_name1  = "csvtest/20141201231852094000_Avg.csv"
    self._file_name2  = "csvtest/validation_set.csv"
    self._output_file = "lol.csv"
    self._spam1 = parser(self._file_name1)
    self._spam2 = parser(self._file_name2)
    self.foo = self.centered

  def build(self):
    self._build_X_from_files()

  ####################################################################################
  ## INER FUNCTIONS

  def _build_X_from_files(self):
    count = 0
    for row1 in self._spam1:
      row2 = self._spam2.next()
      count += 1
      if count == 1:
        continue
      self.X.append((float(row1[1]),int(row2[1])))
    self.X.sort(key=lambda x : x[0])

  ####################################################################################
  ## RUNNING FUNCTIONS

  def run(self):
    self._build_X_from_files()
    density = 0
    self.result = [(0.0,0.0)]
    for c,x in enumerate(self.X):
      self.result.append((x[0],self.foo(c,self.X,1000)))
    self.result.append((1.0,1.0))
    self.write()


  def write(self):
    b = open(self._output_file, 'w')
    a = csv.writer(b)
    a.writerows(self.result)
    b.close()

  ####################################################################################
  ## DENSITY FUNCTIONS

  def classic(self, index, vector,k=100):
    s = 0
    limit = False
    for i in xrange(k):
      try:
        s += vector[index - 1 + i][1]
      except:
        s = s/float(i)
        limit = True
        break
    if not limit:
      s = s/float(k)
    return s

  def centered(self, index, vector, k=100):
    s = 0
    c = 0
    for i in xrange(max(0,index-k/2),min(len(vector),index+k/2)):
      s += vector[i][1]
      c += 1
    s = s/float(c)
    return s
 
  ####################################################################################
  ## EXTERNAL FUNCTIONS

  def adjust(self, value):
    previous = 0
    for c, (proba, click) in enumerate(self.X):
      if value == proba:
        return self.foo(c,self.X,1000)
      elif value < proba:
        return self.foo(c,self.X,1000) #FIXME Aproximation
