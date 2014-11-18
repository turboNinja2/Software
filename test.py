#!/usr/bin/env python

from settings               import *
from Model                  import *
from Models                 import *
from Globals                import *

MULTI = True


Learning    = OnlineLinearLearning

train       = dataPath + "small_train_set.csv"
validation  = dataPath + "small_validation_set.csv"
dump        = dataPath + "results/test_results.csv"
json_dump   = dataPath + "results/test_json_results.csv" 

kwargs = {
  "trainPath"       : train,
  "validationPath"  : validation,
  "dumpingPath"     : dump,
  "jsonDumpingPath" : json_dump,
  "refreshLine"     : 100,
}


expected    = 0.167259060835432709423848


model = Learning({"alpha" : 0.1}, 
                 [0.] * D, 
                 **kwargs
                 )
model.train()
found = model.validate()
model.dump_score()

if found == expected:
  print("We're Good !")
else:
  print("Houston, we got a problem. Found : %s, Expected : %s" % (found, expected))

if MULTI:
  model_list = []
  for i in range(num_cores):
    model_list.append(OnlineLinearLearning({"alpha" : 10 ** (-i)}, 
                  [0] * D,
                  **kwargs
                  ))

  models = Models(model_list)
  models.train()
  models.validation()
  models.dump()

print("Test ended")

