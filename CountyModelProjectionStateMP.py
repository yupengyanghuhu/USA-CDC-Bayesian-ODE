import pandas as pd
import numpy as np
import numpy.matlib as npm
import scipy.optimize
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import sys
from timeit import default_timer as timer
from datetime import timedelta
import multiprocessing as mp
from core.MCMCPosteriorSamplingFunction import MCMCPosteriorSampling
import core.CountryModel
import core.EpiEquations
import core.PostProcessing
import core.ProcessData
import core.Utils
import core.CostFunction
import core.RunProjections

def rerun(input_dict):

	StorageFolder = 'MCMC_' + input_dict['Sim'] + input_dict['thetaprior'] + input_dict['Model'] + '_data' 
	AcceptedFile = 'AcceptedUniqueParameters_FirstRun_sigma2err_'+str(input_dict['sigma2_err_known'])+'_SigmaPrFact_'+str(input_dict['proposal_variance_factor'])+'_' + input_dict['countyName'] + '.txt'  
	AcceptedUniqueParameters = np.loadtxt(StorageFolder + '/' + AcceptedFile)
	input_dict['initial_sample'] = AcceptedUniqueParameters[-1].tolist()
	input_dict['sigma2_err_known'] = 10
	input_dict['proposal_variance_factor'] = 10e4
	input_dict['Model'] = 'SecondRun'

	MCMCPosteriorSampling(input_dict)


def projection(input_dict, tsteps):

	Sim = input_dict['Sim']
	thetaprior = input_dict['thetaprior']
	Model = input_dict['Model']
	countyName = input_dict['countyName']
	FitStartDay = input_dict['FitStartDay']
	InitialInfections = input_dict['InitialInfections']
	InitialExposedMult = input_dict['InitialExposedMult']
	lockdown_begin = input_dict['LockdownBegin']
	lockdown_duration = input_dict['LockdownDuration']
	countyPopulation = input_dict['Population']
	countyDataFile = input_dict['countyDataFile']
	multiprocessingma = input_dict['multiprocessingma']

	StorageFolder = 'MCMC_' + Sim + thetaprior + Model + '_data' 
	Confirmed, Deaths, Dates = core.ProcessData.ImportCountyCaseData(countyDataFile, FitStartDay)
	fitData = np.vstack([Confirmed, Deaths])
	AcceptedFile = 'AcceptedUniqueParameters_SecondRun_sigma2err_'+str(input_dict['sigma2_err_known'])+'_SigmaPrFact_'+str(input_dict['proposal_variance_factor'])+'_' + countyName + '.txt'

	tmin = 1
	tmax = tsteps
	fitTime = np.linspace(tmin,tmax,tsteps)	
	AcceptedUniqueParameters = np.loadtxt(StorageFolder + '/' + AcceptedFile)
	SimLabel = 'Projection_' + countyName
	core.RunProjections.RunProjections(AcceptedUniqueParameters, tmax, fitData, countyPopulation, InitialInfections, 
		InitialExposedMult, lockdown_begin, lockdown_duration, Dates, SimLabel, multiprocessingma)



def main():
	numproc = mp.cpu_count()
	countyPool = mp.Pool(processes=numproc)
	
	# countyNames = ['Allegany_MD','Anne_Arundel_MD','Baltimore_MD','Calvert_MD','Caroline_MD','Carroll_MD','Cecil_MD','Charles_MD','Dorchester_MD','Frederick_MD','Garrett_MD','Harford_MD','Howard_MD','Kent_MD','Montgomery_MD',"Prince_George's_MD","Queen_Anne's_MD","St_Mary's_MD",'Somerset_MD','Talbot_MD','Washington_MD','Wicomico_MD','Worcester_MD','Baltimore_city_MD']
	countyNames = open('USStatesList.txt', "r").read().split('\n')
	if countyNames[-1] == '':
		countyNames.pop(-1)

	countyPool.map(runMCMCByCounty,countyNames)

		
def runMCMCByCounty(countyName):
	
	print('*********************')
	print('******* Running '+ countyName +' *******')
	for root, dirs, files in os.walk("data/statesData"):
		for fileName in files:
			if countyName in fileName:
				countyPopulation = int(fileName.split('pop')[-1].split('.csv')[0])

				input_dict = {}
				input_dict['Sim'] = ''
				input_dict['thetaprior'] = ''
				input_dict['Model'] = 'FirstRun'
				input_dict['GeogScale'] = 'Global'
				input_dict['Location'] = ['USCounty']
				input_dict['countyName'] = countyName
				input_dict['countyDataFile'] = 'data/statesData/' + fileName
				input_dict['FitStartDay'] = '2020-03-01'
				# input_dict['FitLastDay'] = '2020-06-30'
				input_dict['LockdownBegin'] = 30
				input_dict['LockdownDuration'] = 45
				input_dict['InitialInfections'] = 1
				input_dict['InitialExposedMult'] = 5
				input_dict['iterations'] = 500 #10000
				input_dict['Population'] = countyPopulation
				input_dict['initial_sample'] = [7.16e-01,4.97e-01,1.10e-01,1.21e-01,9.03e-01,3.18e-01,2.06e-01,1.85e-02,4.50e-02,9.83e-01,1.33e-01]
				input_dict['sigma2_err_known'] = 10000 #sigma2_err_known_vec[runID]  # It is the known variance of the error term which is used to compute the log likelihood function, and the likelihood function is based on the error of the fit. if the initial parameters are not really realistic, and start with a higher sigma, it won't accept anything, because it lowers the step size (negative relationship)
				input_dict['proposal_variance_factor'] = 10e2 #proposal_variance_factor_vec[runID] # It is associated with the proposal density which is used to "guess" the parameters in the next iteration, since each guess is based on the proposal density function. the reason it's called proposal is because it is "proposing" a set of parameters to try. allow the tolerance for acceptance, if high, you accept more parameters which not close to the true parameter
				input_dict['multiprocessingma'] = True

				MCMCPosteriorSampling(input_dict)
				
				############## Get Best Fit Parms And Run again ################
				print('Get Best Fit Parms And Run again...')
				rerun(input_dict)

				############## Get Projection ################
				print('Get Projection...')
				projection(input_dict, 500)



if __name__ == "__main__":
	# execute only if run as a script
	main()

