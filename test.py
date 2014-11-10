#!/usr/bin/env python

from settings               import *
from Learn                  import *
from ErrorEvaluation        import *
from Model                  import *
from Models                 import *
from Globals                import *

refreshLineTest = 150
MULTI = False

params = {"alpha" : 0.1}   # learning rate for sgd optimization
w = [0.] * D
Learning = OnlineLinearLearning

train       = dataPath + "small_train_set.csv"
validation  = dataPath + "small_validation_set.csv"

expected = 0.167259060835432709423848

model = Learning(params, w)
model.run(train,customRefreshLine=refreshLine)
found = model.run(validation,customRefreshLine=refreshLineTest,update=False)

#trainModels(train,[model] * 2)

if found == expected:
  print("We're Good !")
else:
  print("Houston, we got a problem. Found : %s, Expected : %s" % (found, expected))

if MULTI:
  model_list = [Learning(params, w)] * 2
  models = Models(model_list)
  models.train(train)
  models.validation(validation)