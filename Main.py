#!/usr/bin/env python

from settings         import *
from Export           import writeSubmission
from Model            import *
from Models           import *

if __name__ == '__main__':
  import test

  # folders #################################################################
  train_global    = dataPath + 'train_rev2.csv'  # path to training file
  test_global     = dataPath + 'test_rev2.csv'  # path to testing file
  train_set       = dataPath + 'medium_train_set.csv'  # path to training file
  validation_set  = dataPath + 'medium_validation_set.csv'  # path to testing file
  dump            = dataPath + "results/results.csv"


  # training and testing
  # #######################################################

  model_list = []

  for i in range(0,3*num_cores):
    model_list.append(OnlineLinearLearning({"alpha" : pow(10,-1) + i*(pow(10,-3)-pow(10,-1))/(3*num_cores)},
                  [0] * D, 
                  trainPath       = train_set,
                  validationPath  = validation_set,
                  refreshLine     = refreshLine,
                  dumpingPath      = dump,
    ))

  if validation :
    models = Models(model_list)
    models.train()
    models.validation()
    models.dump()
  if submit :
    model = Learning(params,w,train_global)
    model.train()
    writeSubmission(dataPath,model)

  print('Hello World Juju and Ulysse')
