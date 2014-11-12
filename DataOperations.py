from Globals import *
from FeatureFunctions import *

# A.  x, y generator
# INPUT:
#   path: path to train.csv or test.csv
#   label_path: (optional) path to trainLabels.csv
# YIELDS:
#   ID: id of the instance (can also acts as instance count)
#   x: a list of indices that its value is 1
#   y: (if label_path is present) label value of y1 to y33

#########
## TOOLS
def strip_line(line):
  return line.rstrip().split(',')

def hash_feature(m,feat):
  return abs(hash(str(m) + '_' + feat)) % D

def hash_features(m,feats):
  result = []
  for i, feat in enumerate(feats):
    result.append(abs(hash(str(m) + '_' + str(i) + '_' + feat)) % D)
  return result

class DataParser:
 
  ######################################################################
  ## PARSING METHOD
  def classic(self, line, nbZeroes=2):
    x = [0] * nbZeroes
    for m, feat in enumerate(strip_line(line)):
      if m == 0:
        ID = int(feat)
      elif self.traindata and m == 1:
        y = float(feat)
      else:
        x.append(abs(hash(str(m) + '_' + feat)) % D)
    return (ID, x, y)

  def classic2(self, line, nbZeroes=1):
    x = [0] * nbZeroes
    for m, feat in enumerate(strip_line(line)):
      if self.header[m] == "id":
        ID = int(feat)
      elif self.header[m] == "hour":
        x.extend(hash_features(m,cutHourAndDay(feat)))
      elif self.traindata and self.header[m] == "click":
        y = float(feat)
      else:
        x.append(hash_feature(m,feat))
    return (ID, x, y)

  def cross_prod(self, line, x):
    i = 0 
    for m1, feat1 in enumerate(stip_line(line)):
      if m1 == 0:
        ID = int(feat1)
      elif self.traindata and m1 == 1:
        y = float(feat1)
      else:
        for m2, feat2 in enumerate(strip_line(line)):
          if m2 != 0 and not (self.traindata and m2 == 1):
            x[i] = abs(hash(str(m1) + '_' + feat1 + '_' + str(m2) + '_' + feat2)) % D
            i += 1
    return (ID, x, y)
            

  PARSING_METHODS = {
    "classic"     : classic,
    "classic2"     : classic2,
    "cross_prod"  : cross_prod,
  }

  PARSING_LENGHT = {
    "classic"     : 27,
    "classic2"     : 27,
    "cross_prod"  : 27 * 27, 
  }
  ######################################################################
  ## CORE FUNCTIONS

  def __init__(self, path, traindata=True, mode="classic2", nbZeroes=2):
    self.path = path
    self.mode = mode
    self.parsing_method = self.PARSING_METHODS[mode]
    self.parsing_lenght = self.PARSING_LENGHT[mode]
    self.traindata = traindata
    self.header = []
    self.nbZeroes = nbZeroes

  def run(self):
    for t, line in enumerate(open(self.path)):
      if t == 0:
        self.header = strip_line(line)
        continue
      ID, x, y = self.parsing_method(self, line, nbZeroes = self.nbZeroes)
      yield (ID, x, y) if self.traindata else (ID, x)

# The files contains 47 686 525 lines
def countLines(path):
  nbLines = 0
  for t, line in enumerate(open(path)):
    nbLines +=1
  return(nbLines)

def createValidationSet(inputPath,filename,small=False,medium=False):
  size = 4 * pow(10,7)
  if small:
    size = pow(10,2)
  elif medium:
    size = 2 * pow(10,6)
  inputFile = inputPath + filename
  if small:
    inputPath += "small_"
  elif medium:
    inputPath += "medium_"
  with open(inputPath + 'train_set.csv', 'w') as outfileTrain:
    with open(inputPath + 'validation_set.csv', 'w') as outfileValidation:
      for t, line in enumerate(open(inputFile)):
        if t == 0:
          header = line
          outfileTrain.write(header)
          outfileValidation.write(header)
          continue
        if t < size:
          outfileTrain.write(line)
        elif t > (size * 4 / 3):
          return
        else:
          outfileValidation.write(line)
