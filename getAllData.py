import shutil
import os
import sys
import datetime
import logging
import atexit
import getopt
import os.path
import smtplib
from os.path import basename
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate
import pandas as pd
import numpy as np
import json
import xlrd
import csv
import pycountry
import re
import ast
import dropbox

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

nyTimeUrl_county = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv"
nyTimeUrl_states = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv"
populationStateFile = "data/USStatesPop.csv"
USURCounties1 = 'data/USCounties.csv'

def getAllCountiesData():
    USURCounties = pd.read_csv(USURCounties1, error_bad_lines=False, encoding='latin-1')
    USURCounties = USURCounties[['FIPS','Areaname','STNAME','STATEFP','2013 Urban-Rural Code','population','POPDENSITY']]
    USURCounties = USURCounties.rename(columns={"Areaname": "Name"})
    USURCounties = USURCounties.rename(columns={"2013 Urban-Rural Code": "Urban-Rural-Classification"})
    USURCounties = USURCounties.rename(columns={"POPDENSITY": "popdensity"})

    nyTimeSeries = pd.read_csv(nyTimeUrl_county, error_bad_lines=False)
    nyTimeSeries = nyTimeSeries.dropna(subset=['fips'])
    
    nyTimeSeries_cases = pd.pivot_table(nyTimeSeries, values='cases', index=["fips"], columns="date").fillna(0)
    nyTimeSeries_cases = nyTimeSeries_cases.reset_index()
    nyTimeSeries_cases['fips']=nyTimeSeries_cases['fips'].apply(int)
    nyTimeSeries_cases = nyTimeSeries_cases.rename(columns={"fips": "FIPS"})
    final_real_cases = pd.merge(USURCounties, nyTimeSeries_cases, on='FIPS')
    final_real_cases = final_real_cases.set_index('FIPS')
    final_real_cases = final_real_cases[final_real_cases['population'] > 50000]
    # final_real_cases = final_real_cases[final_real_cases['STNAME'] == 'Maryland']
    
    nyTimeSeries_deaths = pd.pivot_table(nyTimeSeries, values='deaths', index=["fips"], columns="date").fillna(0)
    nyTimeSeries_deaths = nyTimeSeries_deaths.reset_index()
    nyTimeSeries_deaths['fips']=nyTimeSeries_deaths['fips'].apply(int)
    nyTimeSeries_deaths = nyTimeSeries_deaths.rename(columns={"fips": "FIPS"})
    final_real_deaths = pd.merge(USURCounties, nyTimeSeries_deaths, on='FIPS')
    final_real_deaths = final_real_deaths.set_index('FIPS')
    final_real_deaths = final_real_deaths[final_real_deaths['population'] > 50000]
    # final_real_deaths = final_real_deaths[final_real_deaths['STNAME'] == 'Maryland']
    
    try:
        os.makedirs('data/countiesData')
    except:
        pass
    
    countyNames = []
    for i in range(final_real_cases.shape[0]):
        if str(final_real_cases.iloc[i,0]) != 'nan':
            countyName = final_real_cases.iloc[i,0].replace(', ', '_').replace(' ', '_').replace('.', '')
            countyNames.append(countyName)
            countyName = countyName + '_pop' + str(int(final_real_cases.iloc[i,4]))

            new_df = pd.DataFrame(final_real_cases.iloc[i,:]).T.iloc[:,5:]
            new_df = new_df.rename(columns={"popdensity": 'date'})
            new_df['date'] = 'cumulative confirmed cases'
            new_df2 = pd.DataFrame(final_real_deaths.iloc[i,:]).T.iloc[:,5:]
            new_df2 = new_df2.rename(columns={"popdensity": 'date'})
            new_df2['date'] = 'cumulative deaths'

            dd = pd.concat([new_df,new_df2],ignore_index=True)  #.groupby('CSA_title').sum()
            dd = dd.set_index('date')
            dd.to_csv('data/countiesData/'+countyName+'.csv')

    with open('USCountiesList.txt', 'w') as f:
        for item in countyNames:
            f.write("%s\n" % item)


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
    
    nyTimeSeries_deaths = pd.pivot_table(nyTimeSeries, values='deaths', index=["state"], columns="date").fillna(0)
    nyTimeSeries_deaths = nyTimeSeries_deaths.reset_index()
    final_real_deaths = pd.merge(populationByState[['state','Pop','Growth']],nyTimeSeries_deaths,on='state')
    final_real_deaths = final_real_deaths.set_index('state')
    final_real_deaths = final_real_deaths.sort_index()
    final_real_deaths = final_real_deaths.reset_index()
    
    try:
        os.makedirs('data/statesData')
    except:
        pass
    
    stateNames = []
    for i in range(final_real_cases.shape[0]):
        if str(final_real_cases.iloc[i,0]) != 'nan':
            stateName = final_real_cases.iloc[i,0].replace(', ', '_').replace(' ', '_').replace('.', '')
            stateNames.append(stateName)
            stateName = stateName + '_pop' + str(int(final_real_cases.iloc[i,1]))

            new_df = pd.DataFrame(final_real_cases.iloc[i,:]).T.iloc[:,2:]
            new_df = new_df.rename(columns={"Growth": 'date'})
            new_df['date'] = 'cumulative confirmed cases'
            new_df2 = pd.DataFrame(final_real_deaths.iloc[i,:]).T.iloc[:,2:]
            new_df2 = new_df2.rename(columns={"Growth": 'date'})
            new_df2['date'] = 'cumulative deaths'

            dd = pd.concat([new_df,new_df2],ignore_index=True)  #.groupby('CSA_title').sum()
            dd = dd.set_index('date')
            dd.to_csv('data/statesData/'+stateName+'.csv')

    with open('USStatesList.txt', 'w') as f:
        for item in stateNames:
            f.write("%s\n" % item)


############### Main ####################
getAllCountiesData()
getAllStatesData()



