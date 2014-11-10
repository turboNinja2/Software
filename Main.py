#!/usr/bin/env python


from Learn                  import *
from OnlineLearningMethods  import OnlineLinearLearning
from Globals                import D
from datetime               import datetime
from DataOperations         import *
from ErrorEvaluation        import *
from settings               import *

# folders #################################################################
train_global    = dataPath + 'train_rev2.csv'  # path to training file
test_global     = dataPath + 'test_rev2.csv'  # path to testing file
train_set       = dataPath + 'train_set.csv'  # path to training file
validation_set  = dataPath + 'validation_set.csv'  # path to testing file

# training and testing #######################################################
alpha = .1   # learning rate for sgd optimization
w = [0.] * D
model = OnlineLinearLearning(alpha,w)


trainModel(validation_set,model)

validationError(validation_set,model)

# createValidationSet(dataPath,'train_rev2.csv')
print('Hello World Juju and Ulysse')
print(train)
