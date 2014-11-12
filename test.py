#!/usr/bin/env python

from settings               import *
from Model                  import *
from Models                 import *
from Globals                import *

MULTI = True

Learning = OnlineLinearLearning

train       = dataPath + "small_train_set.csv"
validation  = dataPath + "small_validation_set.csv"
dump        = dataPath + "results/test_results.csv"


expected = 0.167259060835432709423848

model = Learning({"alpha" : 0.1}, 
                 [0.] * D, 
                 trainPath=train, 
                 validationPath=validation, 
                 refreshLine=200)
model.train()
found = model.validate()
model.dump_score(dump)

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
                  refreshLine=100))

  models = Models(model_list)
  models.train()
  models.validation()

print("Test ended")

"""
class Caca:
  def __init__(self, bite):
    self.inner = bite

def modify_caca(test_caca):
  test_caca.inner = "couille"

zizi = Caca("chatte")
print(zizi.inner)
modify_caca(zizi)
print(zizi.inner)
print(" test ok")
"""
