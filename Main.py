#!/usr/bin/env python

from settings         import *
from main_settings    import *
from Model            import *
from Models           import *
from Export import *

if __name__ == '__main__':
  if validation :
    models = Models(model_list)
    models.train()
    models.validation()
    models.dump()
  if submit :
    model.train()
    print model.trainPath
    print model.submissionPath
    model.writeSubmission()

  print('Hello World Juju and Ulysse')
