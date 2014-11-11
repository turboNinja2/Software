#!/usr/bin/env python

from settings import *
from DataOperations import *


initial_data = "train_rev2.csv"

"""
createValidationSet(dataPath, initial_data)
createValidationSet(dataPath, initial_data,True)
"""
createValidationSet(dataPath, initial_data,False,True)


