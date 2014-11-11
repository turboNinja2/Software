#!/usr/bin/env python

from settings               import *
from Model                  import *
from Models                 import *
from Globals                import *

MULTI = False

params = {"alpha" : 0.1}   # learning rate for sgd optimization
#params = {"delta" : 0.1, "rho" : 0.1}
w = [0.] * D
Learning = OnlineLinearLearning
#Learning = ZALMS

train       = dataPath + "small_train_set.csv"
validation  = dataPath + "small_validation_set.csv"

expected = 0.167259060835432709423848

model = Learning(params, w, trainPath=train, validationPath=validation, refreshLine=150)
model.train()
found = model.validate()


if found == expected:
  print("We're Good !")
else:
  print("Houston, we got a problem. Found : %s, Expected : %s" % (found, expected))

if MULTI:
  model_list = [Learning(params, w)] * 2
  models = Models(model_list)
  models.train(train)
  models.validation(validation)
