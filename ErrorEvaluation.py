from math import log, exp
from OnlineLearningMethods import OnlineLinearLearning
from DataOperations import data

# B. Bounded logloss
# INPUT:
#     p: our prediction
#     y: real answer
# OUTPUT
#     bounded logarithmic loss of p given y
def logloss(p, y):
    p = max(min(p, 1. - 10e-15), 10e-15)
    return -log(p) if y == 1. else -log(1. - p)

def validationError(validationPath,model):
    for ID, x, y in data(validationPath, traindata = True):
        p = model.predict(x)
        loss += logloss(p, y)  

        if tt % 100000 == 0:
            print('%s\tencountered: %d\tcurrent logloss: %f' % (
                    datetime.now(), tt, (loss * 1./tt)))
        tt += 1
    return (loss * 1./tt)