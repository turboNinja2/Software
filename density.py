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
    self.computed = False

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
    print "Model extracted from files"

  def _compute(self):
    self.result = [(0.0,0.0)]
    self.dico   = {}
    self.dico[0.0] = 0.0
    for c,x in enumerate(self.X):
      temp = (x[0],self.foo(c,self.X,1000))
      self.result.append(temp)
      self.dico[temp[0]] = temp[1]
    self.result.append((1.0,1.0))
    self.computed = True
    print "model computed"
 
  def _write(self):
    b = open(self._output_file, 'w')
    a = csv.writer(b)
    a.writerows(self.result)
    b.close()
    print "result wrote on files"

  ####################################################################################
  ## RUNNING FUNCTIONS

  def run(self,write=False):
    self._build_X_from_files()
    self._compute()
    if write:
      self._write()

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
    return self.dico[value]
    for c, (proba, click) in enumerate(self.X):
      if value == proba:
        if self.computed:
          return self.result[c]
        else:
          return self.foo(c,self.X,1000)
      elif value < proba:
        return self.foo(c,self.X,1000) #FIXME Aproximation
    i = 0
    past = self.X[0]
    while(True):
      if value > self.X[i]:
        

