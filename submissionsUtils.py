import pandas as pd

def average(path,files):
  nbFiles = len(files)
  pdFiles = [] 
  for i in range(nbFiles):
    pdFiles.append(pd.read_csv(path + files[i]))

dataPath = 'C:/Users/JUJulien/Desktop/KAGGLE/Competitions/Avazu/Data/'
submissionPath = dataPath + 'submissions/'

average(submissionPath,['submission20141123011212092878.csv','submission20141123011212092878.csv'])