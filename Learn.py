from Globals import *
from math import log, exp, sqrt
from ErrorEvaluation import logloss
from datetime import datetime
from OnlineLearningMethods import *
from DataOperations import data
from multiprocessing import Pool

def trainModel(trainPath,model):
    n = [0.] * D
    loss = 0.
    tt = 1
    
    for ID, x, y in data(trainPath, traindata = True):
        p = model.predict(x)
        model.update(n, x, y)
        loss += logloss(p, y)  # for progressive validation

        # print out progress, so that we know everything is working
        if tt % refreshLine == 0:
            print('%s\tencountered: %d\tcurrent logloss: %f' % (datetime.now(), tt, (loss * 1. / tt)))
        tt += 1

def trainModels(trainPath,models):
    n = [0.] * D
    tt = 1
    nbModels = len(models)
    losses = [0.] * nbModels 

    pool = Pool(processes=nbCpus)

    def printError(tt):
        time = datetime.now()
        for i in range(0,nbModels):
            print('%s\tencountered: %d\tcurrent logloss: %f' % (time, tt, (losses[i] * 1. / tt)))

    for ID, x, y in data(trainPath, traindata = True):
        def updateModel(i):
            models[i].update(n,x,y)
        
        def evaluateError(i):
            p = models[i].predict(x)
            losses[i] += logloss(p, y) 

        pool.map(updateModel, range(0,nbModels))
        pool.map(evaluateError, range(0,nbModels))

        # print out progress, so that we know everything is working
        if tt % refreshLine == 0:
            printError(tt)

        tt += 1