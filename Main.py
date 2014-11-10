#!/usr/bin/env python
from Learn          import *
from ErrorEvaluation    import *
from settings         import *
from Export import writeSubmission
from OnlineLearningMethods import *
import test

# folders #################################################################
train_global  = dataPath + 'train_rev2.csv'  # path to training file
test_global   = dataPath + 'test_rev2.csv'  # path to testing file
train_set     = dataPath + 'train_set.csv'  # path to training file
validation_set  = dataPath + 'validation_set.csv'  # path to testing file

# training and testing #######################################################

models = [ZALMS([0.001,0.001], [0]*D),  OnlineLinearLearning([0.1], [0]*D)]

if validation :
  # model = Learning(params,w)
  trainModels(train_set,models)
  validationError(validation_set,model)
if submit :
  model = Learning(params,w)
  trainModel(train_global,model)
  writeSubmission(dataPath,model)

print('Hello World Juju and Ulysse')
