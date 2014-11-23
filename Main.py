#!/usr/bin/env python

from settings         import *
from main_settings    import *
from Model            import *
from Models           import *

if __name__ == '__main__':
  if validation :
    models = Models(model_list)
    models.train()
    models.validation()
    models.dump()
  if submit :
    model.train()
    writeSubmission(dataPath,model)

  print('Hello World Juju and Ulysse')
