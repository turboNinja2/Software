#!/usr/bin/env python

from settings         import *
from main_settings    import *
from CustomModels     import *
from Models           import *

if __name__ == '__main__':
  if validationBool :
    models = Models(model_list)
    models.train()
    models.validation()
    models.dump()
    if multipleSubmissions :
      models.writeSubmissions()
  if submitBool :
    model.train()
    model.validate()
    print(model.trainPath)
    print(model.submissionPath)
    model.writeSubmission()

  print('Hello World Juju and Ulysse')