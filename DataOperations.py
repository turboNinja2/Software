from Globals          import *
from FeatureFunctions import *
from random import *

########################################################################
## TOOLS
def strip_line(line):
  return line.rstrip().split(',')

def hash_feature(nameFeat,feat):
  return abs(hash(nameFeat + '_' + feat)) % D

def hashVect(vect):
  n = len(vect)
  res = [0] * n
  for i in range(n):
    res[i] = abs(hash(vect[i])) % D
  return res

def utilCrossProd(vectString1, vectString2):
  n = len(vectString1)
  res = [''] * (n * (n + 1) / 2)
  index = 0
  for i in range(n) :
    for j in range(i,n) :
      res[index] = vectString1[i] + vectString2[j]
      index = index + 1
  return res

def utilCrossProdPure(vectString1, vectString2):
  n = len(vectString1)
  res = [''] * (n * (n - 1) / 2)
  index = 0
  for i in range(n) :
    for j in range(i + 1,n) :
      res[index] = vectString1[i] + vectString2[j]
      index = index + 1
  return res

class DataParser:
 
  ######################################################################
  ## PARSING METHOD

  def base(self,line):
    xString = []
    y = 0
    header = self.header
    for m, feat in enumerate(strip_line(line)):
      if header[m] == "id":
        ID = int(feat)
      elif header[m] == "hour":
        xString.extend(cutHourAndDay(feat))
      elif header[m] == "click":
        y = float(feat)
      else:
        xString.append(header[m] + '_' + feat)
    return(ID,xString,y)

  def classic(self, line):
    y = 0
    x = [0]
    for m, feat in enumerate(strip_line(line)):
      if m == 0:
        ID = int(feat)
      elif self.traindata and m == 1:
        y = float(feat)
      else:
        x.append(abs(hash(str(m) + '_' + feat)) % D)
    return (ID, x, y)

  def classic2(self, line):
    ID,xString,y = self.base(line)
    xString.append('0') # the constant
    x = hashVect(xString)
    return (ID, x, y)

  def crossProdPure(self,line):    
    ID,xString,y = self.base(line)
    xString = utilCrossProdPure(xString,xString)
    xString.append('0') # the constant
    x = hashVect(xString)
    return (ID, x, y)

  def crossProd(self,line):    
    ID,xString,y = self.base(line)
    xString = utilCrossProd(xString,xString)
    xString.append('0') # the constant
    x = hashVect(xString)
    return (ID, x, y)

  PARSING_METHODS = {
    "classic"     : classic,
    "classic2"    : classic2,
    "crossProd" : crossProd,
    "crossProdPure" : crossProdPure
  }


  ######################################################################
  ## CORE FUNCTIONS

  def __init__(self, path, traindata=True, mode="classic"):
    self.path = path
    self.mode = mode
    self.parsing_method = self.PARSING_METHODS[mode]
    self.traindata = traindata
    self.header = []

  def run(self):
    for t, line in enumerate(open(self.path)):
      if t == 0:
        self.header = strip_line(line)
        continue
      ID, x, y = self.parsing_method(self, line)
      yield (ID, x, y) 

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

def createRandomSet(inputPath,filename,selectedSeed) :
  seed(selectedSeed * 123456789)
  inputFile = inputPath + filename
  with open(inputPath + 'train_seed' + str(selectedSeed ) + '.csv', 'w') as randomizedFile:
    for t, line in enumerate(open(inputFile)):
      if t == 0:
        header = line
        randomizedFile.write(header)
        continue
      if random() > 0.7:
        randomizedFile.write(line)