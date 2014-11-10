#!/usr/bin/env python
from Learn          import *
from ErrorEvaluation    import *
from settings         import *
from Export import writeSubmission
from OnlineLearningMethods import *

if __name__ == '__main__':

  import test

  # folders #################################################################
  train_global = dataPath + 'train_rev2.csv'  # path to training file
  test_global = dataPath + 'test_rev2.csv'  # path to testing file
  train_set = dataPath + 'train_set.csv'  # path to training file
  validation_set = dataPath + 'validation_set.csv'  # path to testing file

  # training and testing
  # #######################################################

  models = []
  for i in range(10):
    models.append(OnlineLinearLearning([5 ** -i], [0] * D))

  if validation :
    # model = Learning(params,w)
    trainModels(train_set,models)
    validationErrors(validation_set,models)
  if submit :
    model = Learning(params,w)
    trainModel(train_global,model)
    writeSubmission(dataPath,model)

  print('Hello World Juju and Ulysse')
