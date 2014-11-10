from datetime import datetime
from OnlineLearningMethods import OnlineLinearLearning

def writeSubmission(submissionPath,model):
    exportTime = datetime.now()
    
    submissionName = exportTime +'_Submission.csv' 
    descriptionName = exportTime + '_Description.txt'

    with open(submissionPath + submissionName, 'w') as outfile:
        outfile.write('id,click\n')
        for ID, x in data(test):
            p = model.predict(x)
            outfile.write('%s,%s\n' % (ID, str(p)))
    
    with open(submissionPath + descriptionName, 'w') as outfile:
        outfile.write(model.description())

    print('Done, elapsed time: %s' % str(datetime.now() - start))
