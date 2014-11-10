#!/usr/bin/env python


from settings         import *
from Learn            import *
from ErrorEvaluation  import *



train       = dataPath + "small_train_set.csv"
validation  = dataPath + "small_validation_set.csv"


expected = 0.167259060835432709423848


model = Learning(params, w)
trainModel(train, model)
found = validationError(validation, model)


if found == expected:
  print "We're Good !"
else:
  print "Houston, we got a problem. Found : %s, Expected : %s" % (found, expected)
