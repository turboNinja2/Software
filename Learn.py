from Globals import *
from math import log, exp, sqrt
from ErrorEvaluation import logloss
from datetime import datetime
from OnlineLearningMethods import *
from DataOperations import *
from multiprocessing import Pool
from joblib import Parallel, delayed  
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
        
