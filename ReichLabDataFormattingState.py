import pandas as pd
import numpy as np
import os
import sys
# from datetime import datetime
import datetime
from datetime import timedelta
import random

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# nyTimeUrl_state = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv"
# USURCounties1 = 'data/USCounties.csv'
nyTimeUrl_states = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv"
populationStateFile = "data/USStatesPop.csv"

def getAllStatesData():
    populationByState = pd.read_csv(populationStateFile, error_bad_lines=False, encoding='latin-1')
    populationByState['state'] = populationByState['State']

    nyTimeSeries = pd.read_csv(nyTimeUrl_states, error_bad_lines=False)
    nyTimeSeries_cases = pd.pivot_table(nyTimeSeries, values='cases', index=["state"], columns="date").fillna(0)
    nyTimeSeries_cases = nyTimeSeries_cases.reset_index()
    final_real_cases = pd.merge(populationByState[['state','Pop','Growth']],nyTimeSeries_cases,on='state')
    final_real_cases = final_real_cases.set_index('state')
    final_real_cases = final_real_cases.sort_index()
    final_real_cases = final_real_cases.reset_index()
    
    stateNames = []
    for i in final_real_cases['state'].tolist():
        stateNames.append(i.replace(', ', '_').replace(' ', '_').replace('.', ''))
    
    return stateNames
 
def ReichLabDataFormatting(time_folder, stateNames, selection, walkFolder):

    # setup(time_folder)

    allStatedflist = []

    for state in stateNames:
        for root, dirs, files in os.walk(walkFolder, topdown=False):
            for filename in files:
                if selection in filename and state in filename:
                    df = pd.read_csv(walkFolder + "/" + filename, header = None)
                    for i in range(len(days_ahead) - 1): 

                        time_next = datetime.datetime.fromisoformat(time_folder) + datetime.timedelta(days_ahead[i+1])
                        time_next = time_next.strftime('%Y-%m-%d')

                        new_days = days_since_start + days_ahead[i+1]
                        current_days = days_since_start + days_ahead[i]

                        mean = round((df.iloc[new_days,:] - df.iloc[current_days,:]).mean(),2)
                        quantile0025 = round((df.iloc[new_days,:] - df.iloc[current_days,:]).quantile(0.025),2)
                        quantile0100 = round((df.iloc[new_days,:] - df.iloc[current_days,:]).quantile(0.1),2)
                        quantile0250 = round((df.iloc[new_days,:] - df.iloc[current_days,:]).quantile(0.25),2)
                        quantile0500 = round((df.iloc[new_days,:] - df.iloc[current_days,:]).quantile(0.5),2)
                        quantile0750 = round((df.iloc[new_days,:] - df.iloc[current_days,:]).quantile(0.750),2)
                        quantile0900 = round((df.iloc[new_days,:] - df.iloc[current_days,:]).quantile(0.900),2)
                        quantile0975 = round((df.iloc[new_days,:] - df.iloc[current_days,:]).quantile(0.975),2)
                        value = [mean, quantile0025, quantile0100, quantile0250, quantile0500, quantile0750, quantile0900, quantile0975]

                        d = pd.DataFrame(value,columns=['value'])
                        d['forecast_date'] = time_folder
                        d['location'] = state
                        # d['location'] = stateFips[stateNames.index(state)]
                        # d['location'] = d['location'].apply(lambda x: '{0:0>5}'.format(x)).astype(str)
                        d['target'] = str(i+1) + ' wk ahead inc case'
                        d['target_end_date'] = time_next
                        d['type'] = s
                        d['quantile'] = w

                        d = d[order]

                        allStatedflist.append(d)

    final_df = pd.concat(allStatedflist)
    # final_df = final_df[final_df['location'] != '51161']
    final_df = final_df.set_index('location')
    final_df.to_csv(time_folder + '-' + selection + '-CDDEP-SEIR.csv')

# def setup(time_folder):
#     start_day = '2020-01-22'
#     days_ahead = [0,7-1,14-1,21-1,28-1,35-1,42-1,49-1,56-1]
#     quantiles = [0.025, 0.100, 0.250, 0.500, 0.750, 0.900, 0.975]
#     order = ["location","target","type","quantile","forecast_date","target_end_date","value"]
#     s = ['point']
#     s.extend(['quantile']*len(quantiles))
#     w = ['NA']
#     w.extend(quantiles)
#     days_since_start = datetime.datetime.strptime(time_folder, "%Y-%m-%d") - datetime.datetime.strptime(start_day, "%Y-%m-%d")
#     days_since_start = days_since_start.days


#################### Main ####################
start_day = '2020-01-22'
time_folder = '2020-07-15'
days_ahead = [0-1,7-1,14-1,21-1,28-1,35-1,42-1,49-1,56-1]
quantiles = [0.025, 0.100, 0.250, 0.500, 0.750, 0.900, 0.975]
order = ["location","target","type","quantile","forecast_date","target_end_date","value"]
s = ['point']
s.extend(['quantile']*len(quantiles))
w = ['NA']
w.extend(quantiles)
days_since_start = datetime.datetime.strptime(time_folder, "%Y-%m-%d") - datetime.datetime.strptime(start_day, "%Y-%m-%d")
days_since_start = days_since_start.days

selections = ['Cumulative Total Infections', 'Cumulative Hospitalized', 'Deaths']
walkFolders = ['results_CumTotInf_07_15', 'results_CumHosp_07_15', 'results_Deaths_07_15']

stateNames = getAllStatesData()
ReichLabDataFormatting(time_folder, stateNames, selections[0], walkFolders[0])
ReichLabDataFormatting(time_folder, stateNames, selections[1], walkFolders[1])
ReichLabDataFormatting(time_folder, stateNames, selections[2], walkFolders[2])




