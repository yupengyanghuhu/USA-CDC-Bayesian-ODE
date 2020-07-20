import pandas as pd
import numpy as np
import os

def ImportCountyCaseData(countyDataFile, startDate):
    Data = pd.read_csv(countyDataFile)

    Dates = Data.columns
    Dates = Dates.drop(['date'])

    ConfirmedCases = Data[Data["date"].isin(['cumulative confirmed cases'])]
    DeathsCases = Data[Data["date"].isin(['cumulative deaths'])]
    ConfirmedCases = ConfirmedCases.drop(['date'], axis=1)
    DeathsCases = DeathsCases.drop(['date'], axis=1)

    Confirmed = np.array(ConfirmedCases.values)
    Deaths = np.array(DeathsCases.values)

    endDate = Data.columns[-1]

    # get rid of leading zeros
    dropIndexes = []
    i = 0
    x = 0
    y = 0
    for i in range(len(Confirmed[0])):
        if Dates[i] == endDate or y > 0:
            dropIndexes = [*dropIndexes, i]
            y += 1
        elif Dates[i] == startDate or x == 0:
            dropIndexes = [*dropIndexes, i]
            x += 1

    ConfirmedOut = []
    DeathsOut = []
    for i in range(len(Confirmed)):
        ConfirmedOut = [*ConfirmedOut, np.delete(Confirmed[i], dropIndexes)]
        DeathsOut = [*DeathsOut, np.delete(Deaths[i], dropIndexes)]
    
    Dates = Dates.drop(Dates[dropIndexes])
    ConfirmedOut = np.array(ConfirmedOut)
    DeathsOut = np.array(DeathsOut)
    return ConfirmedOut, DeathsOut, Dates