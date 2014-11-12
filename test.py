#!/usr/bin/env python

from settings               import *
from Model                  import *
from Models                 import *
from Globals                import *

MULTI = True

params = {"alpha" : 0.1}   # learning rate for sgd optimization
#params = {"delta" : 0.1, "rho" : 0.1}
w = [0.] * D
Learning = OnlineLinearLearning
#Learning = ZALMS

train       = dataPath + "small_train_set.csv"
validation  = dataPath + "small_validation_set.csv"
dump        = dataPath + "../results/test_results.csv"

expected = 0.167259060835432709423848

model = Learning(params, w, trainPath=train, validationPath=validation, refreshLine=150)
model.train()
found = model.validate()
model.dump_score(dump)

if found == expected:
  print("We're Good !")
else:
  print("Houston, we got a problem. Found : %s, Expected : %s" % (found, expected))

if MULTI:
  model_list = []
  for i in range(5):
    model_list.append(
      OnlineLinearLearning(
                  {"alpha" : 5 ** -i}, 
                  [0] * D, 
                  trainPath=train,
                  validationPath=validation,
                  refreshLine=150
    )
  )

  models = Models(model_list)
  models.train()
  models.validation()
