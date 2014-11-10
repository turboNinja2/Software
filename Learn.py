from Globals import *
from settings import *
from math import log, exp, sqrt
from ErrorEvaluation import logloss
from datetime import datetime
from Model import *
from DataOperations import *
from multiprocessing import Pool
from joblib import Parallel, delayed
import multiprocessing


def update(model, path):
  return model.update(path)

def trainModels(trainPath,models):
    pool = Parallel(n_jobs = num_cores)
    pool(delayed(update)(model,trainPath) for model in models)
