from Globals import *
from math import log, exp, sqrt
from ErrorEvaluation import logloss
from datetime import datetime
from OnlineLearningMethods import *
from DataOperations import *
from multiprocessing import Pool

def trainModel(trainPath,model):
    n = [0.] * D
    loss = 0.
    tt = 1
    data = DataParser(trainPath) 

    for ID, x, y in data.run():
        p = model.predict(x)
        model.update(n, x, y)
        loss += logloss(p, y)  # for progressive validation
        # print out progress, so that we know everything is working
        if tt % refreshLine == 0:
         print('%s\tencountered: %d\tcurrent logloss: %f' % (datetime.now(), tt, (loss * 1. / tt)))
        tt += 1

