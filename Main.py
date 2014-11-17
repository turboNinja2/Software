#!/usr/bin/env python

from settings         import *
from Export           import writeSubmission
from Model            import *
from Models           import *
from datetime import datetime


if __name__ == '__main__':
  import test

  # folders #################################################################
  train_global = dataPath + 'train_rev2.csv'  # path to training file
  test_global = dataPath + 'test_rev2.csv'  # path to testing file
  train_set = dataPath + 'train_set.csv'  # path to training file
  validation_set = dataPath + 'validation_set.csv'  # path to testing file
  dumping_file = dataPath + 'dump' + datetime.now().__str__() +'.csv'  # path to testing file

  # training and testing
  # #######################################################

  model_list = []

  for j in range(0,6):
    for i in range(1,6):
      model_list.append(OLBI({"delta" : 0.1 * 10 ** (-j), "gamma" : 10 ** (-i - j - 1),
                    "features" : "classic2" }, 
                    [0] * D, 
                    trainPath=train_set,
                    validationPath=validation_set,
                    refreshLine=refreshLine,
                    dumpingPath=dumping_file))

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
