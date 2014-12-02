from FeatureFunctions import *
from settings import dataPath 


def strip_line(line):
  return line.rstrip().split(',')

def base(header,line):
    xString = []
    y = 0
    for m, feat in enumerate(strip_line(line)):
      if header[m] == "id":
        ID = int(feat)
      elif header[m] == "click":
        y = float(feat)
      else:
        xString.append(feat)
    return(ID,xString,y)

def countFeatures(filePath) :
  for t, line in enumerate(open(filePath)):
    if t == 0:
      header = strip_line(line)
      continue
    if t == 1:
      ID, xString, y = base(header, line)
      p = len(xString)
      dictionnaries = [{}] * p
      for j in range(p):
        dictionnaries[j] = {}
        dictionnaries[j][xString[j]] = 1
    else:
      ID, xString, y = base(header, line)
      for j in range(p):
        if dictionnaries[j].has_key(xString[j]):
          dictionnaries[j][xString[j]] += 1
        else:
          dictionnaries[j][xString[j]] = 1


def reduceFile(filePath) :
  for t, line in enumerate(open(filePath)):
    if t == 0:
      header = strip_line(line)
      continue
    if t == 1:
      ID, xString, y = base(header, line)
      p = len(xString)
      dictionnariesInverse = [{}] * p
      for j in range(p):
        dictionnariesInverse[j] = {}
        dictionnariesInverse[j][xString[j]] = 0
        print(j)
    else:
      ID, xString, y = base(header, line)
      for j in range(p):
        if not dictionnariesInverse[j].has_key(xString[j]):
          dictionnariesInverse[j][xString[j]] = len(dictionnariesInverse[j])
      if(t % 1000000 == 0):
        print(t)

  print("Inverse dictionnary created")

  with open(filePath + 'reduced', 'w') as reducedFile:
    for t, line in enumerate(open(filePath)):
      if t == 0:
        header = strip_line(line)
        reducedFile.write(header)
        continue
      else:
        ID, xString, y = base(header, line)
        line = y
        for j in range(1,len(xString)) :
          line = line + ';' + dictionnariesInverse[j][xString[j]]
          reducedFile.write(line)
          if(t % 1000000 == 0):
            print(t)


trainGlobal = dataPath + 'train.csv'
reduceFile(trainGlobal)
