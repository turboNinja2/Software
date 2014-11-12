#!/usr/bin/env python

from settings               import *
from Model                  import *
from Models                 import *
from Globals                import *

MULTI = True

Learning = OnlineLinearLearning

train = dataPath + "small_train_set.csv"
validation = dataPath + "small_validation_set.csv"
dump = dataPath + "Results/test_results.csv"


expected = 0.167259060835432709423848


model = Learning({"alpha" : 0.1}, 
                 [0.] * D, 
                 trainPath=train, 
                 validationPath=validation, 
                 refreshLine=100,
                 dumpingPath=dump)
model.train()
found = model.validate()
model.dump_score()

if found == expected:
  print("We're Good !")
else:
  print("Houston, we got a problem. Found : %s, Expected : %s" % (found, expected))

if MULTI:
  model_list = []
  for i in range(6):
    model_list.append(OnlineLinearLearning({"alpha" : 10 ** (-i)}, 
                  [0] * D, 
                  trainPath=train,
                  validationPath=validation,
                  refreshLine=50,
                  dumpingPath=dump))

  models = Models(model_list)
  models.train()
  models.validation()
  models.dump()

print("Test ended")

