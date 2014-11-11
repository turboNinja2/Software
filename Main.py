#!/usr/bin/env python

from settings         import *
from Export           import writeSubmission
from Model            import *
from Models           import *

if __name__ == '__main__':
  import test

  # folders #################################################################
  train_global = dataPath + 'train_rev2.csv'  # path to training file
  test_global = dataPath + 'test_rev2.csv'  # path to testing file
  train_set = dataPath + 'train_set.csv'  # path to training file
  validation_set = dataPath + 'validation_set.csv'  # path to testing file

  # training and testing
  # #######################################################

  model = PA({}, [0] * D)

  model_list = []

  for i in range(5):
    model_list.append(OnlineLinearLearning(
                  {"alpha" : 5 ** -i}, 
                  [0] * D, 
                  trainPath=train_set,
                  validationPath=validation_set,
                  refreshLine=refreshLine
    ))

  if validation :
    models = Models(model_list)
    models.train()
    models.validation()
  if submit :
    model = Learning(params,w,train_global)
    model.train()
    writeSubmission(dataPath,model)

  print('Hello World Juju and Ulysse')
