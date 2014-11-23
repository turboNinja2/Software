#!/usr/bin/env python

from settings         import *
from Export           import writeSubmission
from Model            import *
from Models           import *
from datetime import datetime

if __name__ == '__main__':
  import l_test
  
  # folders #################################################################
  train_global    = dataPath + 'train_rev2.csv'  # path to training file
  test_global     = dataPath + 'test_rev2.csv'  # path to testing file
  train_set       = dataPath + 'train_set_shuffled.csv'  # path to training file
  validation_set  = dataPath + 'validation_set_shuffled.csv'  # path to testing file
  
  dt = datetime.now().__str__()
  dummyString = ''.join(e for e in dt if e.isalnum())
  dump            = dataPath + "results/results" + dummyString+ ".csv"
  jsonDump            = dataPath + "results/results_json" + dummyString+ ".csv"

  # training and testing
  # #######################################################

  model_list = []


  for i in range(1,6):
    model_list.append(OnlineLinearLearning({"alpha" : 0.1 * 10 ** (-i)}, 
      [0] * D,
      trainPath=train_set,
      validationPath=validation_set,
      refreshLine=refreshLine,
      dumpingPath=dump,
      jsonDumpingPath = jsonDump))


  if validation :
    """
    model = OnlineLinearLearning({"alpha":0.01},[0.]*D,
      trainPath=train_set,
      validationPath=validation_set,
      refreshLine=refreshLine,
      dumpingPath=dump,
      jsonDumpingPath = jsonDump)
    model.train()
    print(model.validate())
    model.dump_score()
    """
    models = Models(model_list)
    models.train()
    models.validation()
    models.dump()
  if submit :
    model = Learning(params,w,train_global)
    model.train()
    writeSubmission(dataPath,model)

  print('Hello World Juju and Ulysse')
