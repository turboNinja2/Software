from Globals import D

# A. x, y generator
# INPUT:
#     path: path to train.csv or test.csv
#     label_path: (optional) path to trainLabels.csv
# YIELDS:
#     ID: id of the instance (can also acts as instance count)
#     x: a list of indices that its value is 1
#     y: (if label_path is present) label value of y1 to y33
def data(path, traindata=False):
    for t, line in enumerate(open(path)):
        if t == 0:
            x = [0] * 27
            continue
        for m, feat in enumerate(line.rstrip().split(',')):
            if m == 0:
                ID = int(feat)
            elif traindata and m == 1:
                y = float(feat)
            else:
                x[m] = abs(hash(str(m) + '_' + feat)) % D

        yield (ID, x, y) if traindata else (ID, x)

# The files contains 47 686 525 lines
def countLines(path):
    nbLines =0
    for t, line in enumerate(open(path)):
        nbLines +=1
    return(nbLines)

def createValidationSet(inputPath,filename):
    inputFile = inputPath + filename
    with open(inputPath + 'train_set.csv', 'w') as outfileTrain:
        with open(inputPath + 'validation_set.csv', 'w') as outfileValidation:
            for t, line in enumerate(open(inputFile)):
                if t == 0:
                    header = line
                    outfileTrain.write(header)
                    outfileValidation.write(header)
                    continue
                if t < 40000000:
                    outfileTrain.write(line)
                else:
                    outfileValidation.write(line)
