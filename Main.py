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

  if validationBool :
    models2 = Models(model_list2)
    models2.train()
    models2.validation()
    models2.dump()
    if multipleSubmissions :
      models2.writeSubmissions()


  print('Hello World Juju and Ulysse')