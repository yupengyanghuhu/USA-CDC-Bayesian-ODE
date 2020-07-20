import numpy as np
import core.ParameterSet
import scipy.stats


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def WriteSeriesCsv(outputData, Compartment, Dir, SimLabel):
    print('Writing ' + Compartment + 'Results')
    np.savetxt(Dir + '/'+  Compartment + SimLabel + '.csv', outputData, delimiter=",", fmt='%5s')
    
    # dataname = ['mean','lower','upper','best']
    # alldata = np.zeros([len(outputData[:,0]),len(dataname)])
    # # # alldata[:,0] = outputData.mean(1)
    # # print(alldata[:,0])
    # for i in range(len(outputData[:,0])):
    #     alldata[i,0], alldata[i,1], alldata[i,2] = mean_confidence_interval(outputData[i,:])

    # alldata[:,3] = outputData[:,-1]
    # np.savetxt(Dir + '/'+ Compartment + SimLabel + '_MeanUpperLower.csv', np.vstack([dataname,alldata]), delimiter=",", fmt='%5s')
