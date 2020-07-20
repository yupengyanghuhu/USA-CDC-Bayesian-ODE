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

nyTimeUrl_county = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv"
USURCounties1 = 'data/USCounties.csv'

def getAllCountyInfo():
    USURCounties = pd.read_csv(USURCounties1, error_bad_lines=False, encoding='latin-1')
    USURCounties = USURCounties[['FIPS','Areaname','STNAME','STATEFP','2013 Urban-Rural Code','population','POPDENSITY']]
    USURCounties = USURCounties.rename(columns={"Areaname": "Name"})
    USURCounties = USURCounties.rename(columns={"2013 Urban-Rural Code": "Urban-Rural-Classification"})
    USURCounties = USURCounties.rename(columns={"POPDENSITY": "popdensity"})

    nyTimeSeries = pd.read_csv(nyTimeUrl_county, error_bad_lines=False)
    getAllDatesList = list(sorted(set(nyTimeSeries["date"])))
    nyTimeSeries = nyTimeSeries.dropna(subset=['fips'])
    
    nyTimeSeries_cases = pd.pivot_table(nyTimeSeries, values='cases', index=["fips"], columns="date").fillna(0)
    nyTimeSeries_cases = nyTimeSeries_cases.reset_index()
    nyTimeSeries_cases['fips']=nyTimeSeries_cases['fips'].apply(int)
    nyTimeSeries_cases = nyTimeSeries_cases.rename(columns={"fips": "FIPS"})
    final_real_cases = pd.merge(USURCounties, nyTimeSeries_cases, on='FIPS')
    final_real_cases = final_real_cases.set_index('FIPS')
#     final_real_cases = final_real_cases[final_real_cases['STNAME'] == stateName]
    final_real_cases = final_real_cases[final_real_cases['population'] > 50000]
    final_real_cases = final_real_cases.dropna(how='all')
    
    countyNames = []
    countyFips = []
    for i in range(final_real_cases.shape[0]):
        countyNames.append(final_real_cases.iloc[i,0].replace(', ', '_').replace(' ', '_').replace('.', ''))
        countyFips.append(final_real_cases.reset_index().iloc[i,0])
    
    return countyNames, countyFips
 

def ReichLabDataFormatting(time_folder, countyNames, countyFips):
    start_day = '2020-01-22'
    days_ahead = [0-1,7-1,14-1,21-1,28-1,35-1,42-1,49-1,56-1]
    quantiles = [0.025, 0.100, 0.250, 0.500, 0.750, 0.900, 0.975]
    # countyNames = ['Allegany_MD','Anne_Arundel_MD','Baltimore_MD','Calvert_MD','Caroline_MD','Carroll_MD','Cecil_MD','Charles_MD','Dorchester_MD','Frederick_MD','Garrett_MD','Harford_MD','Howard_MD','Kent_MD','Montgomery_MD',"Prince_George's_MD","Queen_Anne's_MD","St_Mary's_MD",'Somerset_MD','Talbot_MD','Washington_MD','Wicomico_MD','Worcester_MD','Baltimore_city_MD']
    # countyFips = [24001, 24003, 24005, 24009, 24011, 24013, 24015, 24017, 24019, 24021, 24023, 24025, 24027, 24029, 24031, 24033, 24035, 24037, 24039, 24041, 24043, 24045, 24047, 24510]
    # countyNames, countyFips = getAllCountyInfo()
    order = ["location","target","type","quantile","forecast_date","target_end_date","value"]
    s = ['point']
    s.extend(['quantile']*len(quantiles))
    w = ['NA']
    w.extend(quantiles)
    days_since_start = datetime.datetime.strptime(time_folder, "%Y-%m-%d") - datetime.datetime.strptime(start_day, "%Y-%m-%d")
    days_since_start = days_since_start.days

    allMDdflist = []
    resultsFileName = 'Cumulative Total Infections'

    for county in countyNames:
        for root, dirs, files in os.walk('results_07_12', topdown=False):
            for filename in files:
                if resultsFileName in filename and county in filename:
                    df = pd.read_csv("results_07_12/" + filename, header = None)
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
                        d['location'] = countyFips[countyNames.index(county)]
                        d['location'] = d['location'].apply(lambda x: '{0:0>5}'.format(x)).astype(str)
                        d['target'] = str(i+1) + ' wk ahead inc case'
                        d['target_end_date'] = time_next
                        d['type'] = s
                        d['quantile'] = w

                        d = d[order]

                        allMDdflist.append(d)

    final_df = pd.concat(allMDdflist)
    final_df = final_df[final_df['location'] != '51161']
    final_df = final_df.set_index('location')
    final_df.to_csv(time_folder + '-CDDEP-SEIR.csv')

#################### Main ####################
countyNames, countyFips = getAllCountyInfo()
ReichLabDataFormatting('2020-07-12', countyNames, countyFips)

