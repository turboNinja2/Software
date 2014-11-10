from Globals import *
from math import log, exp, sqrt
from ErrorEvaluation import logloss
from datetime import datetime
from OnlineLearningMethods import *
from DataOperations import *
from multiprocessing import Pool
from joblib import Parallel, delayed
from settings import *
import multiprocessing

def trainModel(trainPath,model):
    tt = 1
    data = DataParser(trainPath) 
    for ID, x, y in data.run():
        model.update(x, y)
        # print out progress, so that we know everything is working
        if tt % refreshLine == 0:
         print('%s\tencountered: %d\tcurrent logloss: %f' % (datetime.now(), tt, model.getLogLoss()))
        tt += 1
 

def foo(model, x, y):
  return model.update(x,y)



def trainModels(trainPath,models):
    tt = 1
    data = DataParser(trainPath)
    nbModels = len(models) 
    pool = Parallel(n_jobs = num_cores)
    for ID, x, y in data.run():

      print pool(delayed(foo)(model,x,y) for model in models)

      if tt % refreshLine == 0:
        print('Encountered: %d\t' % (tt))
        for i in xrange(nbModels):
         print('Description: %s \t logloss: %f ' % (models[i].description(), models[i].getLogLoss()))

      #models[i].update(x, y)
      tt += 1
        
