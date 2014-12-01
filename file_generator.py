#!/usr/bin/env python

from settings       import *
from DataOperations import *

initial_data = "train.csv"

"""
createValidationSet(dataPath, initial_data)
createValidationSet(dataPath, initial_data,True)
createValidationSet(dataPath, initial_data,False,True)
"""

print("started to create files")
for i in range(8):
  print("file : " +str(i))
  createRandomSet(dataPath,initial_data,i)
