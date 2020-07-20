import pandas as pd
import math
import numpy as np
import numpy.matlib as npm
import scipy.optimize
import matplotlib.pyplot as plt
from matplotlib import cm
import os
from timeit import default_timer as timer
from datetime import timedelta
import multiprocessing

import core.CountryModel
import core.EpiEquations
import core.PostProcessing
import core.ProcessData
import core.Utils
import core.CostFunction
import core.RunProjections

"""
    Computes the likelihood of the data given parameter set 'iParams'
"""

def compute_likelihood(iParams, is_sigma2_err_unknown, sigma2_err_known, tmax, fitData, InterventionMult, \
    Population, InitialInfections, InitialExposedMult, lockdown_begin, lockdown_duration):
    
    Residuals = core.CostFunction.ThreeStageSSRFunction(iParams[:-1], tmax, fitData, InterventionMult,\
                    Population, InitialInfections, InitialExposedMult, lockdown_begin, lockdown_duration)

    Residual_sum = np.sum(Residuals)
    if is_sigma2_err_unknown: # sigma2_err is a parameter
        loglikelihood = - (tmax/2)*np.log(2*math.pi*(10**iParams[-1])) - 0.5*(Residual_sum/(10**iParams[-1]))
    else: # sigma2_err constant
        loglikelihood = - (tmax/2)*np.log(2*math.pi*sigma2_err_known) - 0.5*(Residual_sum/sigma2_err_known)
            
    return loglikelihood, Residual_sum

"""
    Defines whether to accept or reject the new sample
"""
def acceptance_rule(pOLD, pNEW):
    
    if pNEW > pOLD:
        return True
    else:
        accept = np.random.uniform(0, 1)
        return (accept < (np.exp(pNEW - pOLD))) # (p1*accept < p2)

"""
    Generating a new sample from uniform proposal density
"""
def transition_model_uniform(x, Difference, lower0, upper0):
    
    lower = x - Difference/2
    for i in range(len(lower)):
        if lower[i] < lower0[i]:
            lower[i] = lower0[i]
    upper = x + Difference/2
    for i in range(len(upper)):
        if upper[i] > upper0[i]:
            upper[i] = upper0[i]
    Difference = upper - lower

    return np.random.rand(1, nDimensions)*Difference + lower

"""
    Generating a new sample from normal proposal density
"""
def transition_model_normal(x, sigma, lower, upper, nDimensions):
    
    x_new = np.random.multivariate_normal(x, sigma, 1)
    for jj in range(nDimensions):
        IsInRange = 0
        while IsInRange == 0:
            IsInRange = 1
            if x_new[0][jj] < lower[jj]:
                IsInRange = 0
                x_new[0][jj] = np.random.normal(x[jj], sigma[jj,jj])
            if x_new[0][jj] > upper[jj]:
                IsInRange = 0
                x_new[0][jj] = np.random.normal(x[jj], sigma[jj,jj])

    return x_new
        
def MCMCPosteriorSampling(input_dict):  # ALL BASED ON SSR, ACCEPT IF THE FIT IS VERY CLOSE TO THE ACTUAL, REJECT IF VERY FAR.

    Sim = input_dict['Sim']
    thetaprior = input_dict['thetaprior']
    Model = input_dict['Model']
    GeogScale = input_dict['GeogScale']
    Location = input_dict['Location']
    countyDataFile = input_dict['countyDataFile']
    countyName = input_dict['countyName']
    FitStartDay = input_dict['FitStartDay']
    # FitLastDay = input_dict['FitLastDay']
    InitialInfections = input_dict['InitialInfections']
    InitialExposedMult = input_dict['InitialExposedMult']
    nDatapointsBuilding_Interp = input_dict['iterations'] # randomly select some number of datapoints
    Population = input_dict['Population']
    multiprocessingma = input_dict['multiprocessingma']

    # print('Starting Posterior Sampling for ' + Sim + ' with ' + thetaprior + ' theta priors')
    StorageFolder = 'MCMC_' + Sim + thetaprior + Model + '_data'
    SimLabel = Sim + thetaprior + Model
    lockdown_begin = input_dict['LockdownBegin']
    lockdown_duration = input_dict['LockdownDuration']
    # Import Case Data
    # Population = 100000
    # Population = 6000000
    # countyDataFile = 'data/MD_data.csv'
    Confirmed, Deaths, Dates = core.ProcessData.ImportCountyCaseData(countyDataFile, FitStartDay)
    # print('Dates')
    # print(Dates)

    numAges = 1
    numLocs = len(Location)
    tmin = 1
    tmax = tsteps = len(Dates)
    fitTime = np.linspace(tmin,tmax,tsteps)
    fitVarsNames = ['alpha', 'beta', 'mu', 'gamma1', 'gamma2', 'gamma3', 'hosrate', 'theta', 'delta', 'lockdown_red', 'postlockdown_red']
    fitData = np.vstack([Confirmed, Deaths])
    lower = [.01, .01, .01, .01, .01, .01, .01, .01, .01, .01, .01] 
    upper = [.99,  2,  .99, .99, .99, .99, .99, .99, .99, .99, .99] 
    mu = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    sigma = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  
    """
        The sigma2_err parameter has EITHER a deterministic value (is_sigma2_err_unknown = False)
        OR has a prior distribution(is_sigma2_err_unknown = True)
        sigma2_err is not a model parameter [error ~ Normal(0,sigma2_err)]
        It is a variance of the regression error
    """
    is_sigma2_err_unknown = False

    """
        The prior for the model parameters can be EITHER uniform (is_prior_normal = False)
        OR normal (is_prior_normal = True)
    """
    is_prior_normal = False

    """
        when sigma2_err is deterministic
    """
    sigma2_err_known = input_dict['sigma2_err_known']

    """
        factor for reducing sigma of proposal w.r.t prior densities
    """
    proposal_variance_factor = input_dict['proposal_variance_factor']

    """
        sigma2_err parameters when it has a prior distribution
    """
    # Adding and sigma for normal prior of sigma2_err
    mu = mu + [0]
    sigma = sigma + [0]
    # shape and scale paramters if sigma2_err has inverse gamma prior
    shape_gamma = 0
    scale_gamma = 0
    # Adding lower and upper bound for sigma2_err
    lower = lower + [0]
    upper = upper + [0]
    
    nDimensions = len(upper)
    lower = np.asarray(lower) 
    upper = np.asarray(upper)
    Difference = upper-lower
    """
        mu and sigma parameters for normal proposal distributions
    """
    mu_proposal = mu
    if is_prior_normal:
        sigma_proposal = [x/proposal_variance_factor for x in sigma]
        sigma_proposal[-1] = sigma[-1]/8
    else:
        sigma_proposal = [x/proposal_variance_factor for x in Difference]
        sigma_proposal[-1] = sigma[-1]/8

    try:
        os.mkdir(StorageFolder)
    except:
        pass



#    x_current = np.random.multivariate_normal(mu_proposal, np.diag(sigma_proposal), 1)
#    x_current = np.array([[0.9027114, 1.479018 , 0.1343914, 0.1401636, 0.7937354, 0.0112248, 10**mu_proposal[-1]]])
    x_current = np.array([input_dict['initial_sample'] + [10**mu_proposal[-1]]])
    # print(x_current[0])
    for jj in range(nDimensions):
        IsInRange = 0
        while IsInRange == 0:
            IsInRange = 1
            if x_current[0][jj] < lower[jj]:
                IsInRange = 0
                x_current[0][jj] = np.random.normal(mu[jj], sigma[jj])
            if x_current[0][jj] > upper[jj]:
                IsInRange = 0
                x_current[0][jj] = np.random.normal(mu[jj], sigma[jj])

    AcceptedUniqueParameters = []
    AcceptedResiduals = []
    RejectedParameters = []
    RejectedResiduals = []
    AcceptCounter = 0
    RejectCounter = 0

    if is_sigma2_err_unknown:
        PosterierParameters = np.zeros([nDatapointsBuilding_Interp, len(lower)])
    else:
        PosterierParameters = np.zeros([nDatapointsBuilding_Interp, len(lower) - 1])
           

    print('total iterations: '+ str(nDatapointsBuilding_Interp))
    for i in range(nDatapointsBuilding_Interp):

        # likelihood for current sample
        parmsFit = [*x_current[0][0:nDimensions-3], x_current[0][-1]]
        InterventionMult = x_current[0][nDimensions-3:nDimensions-1]
        
        x_current_loglik, current_residual = compute_likelihood(parmsFit, is_sigma2_err_unknown, sigma2_err_known, tmax, fitData, InterventionMult, 
                                                                Population, InitialInfections, InitialExposedMult, lockdown_begin, lockdown_duration)


        if i == 0:
            if is_sigma2_err_unknown:
                AcceptedUniqueParameters.append(x_current[0])
            else:
                AcceptedUniqueParameters.append(x_current[0][:-1])
            AcceptedResiduals.append(current_residual)

        if is_prior_normal: # normal prior for model parameters
            normalprior = np.log(multivariate_normal.pdf(x_current[0][:-1], mu[:-1], np.diag(sigma[:-1])))
            posterior_current = x_current_loglik + normalprior # posterior at the current sample
        else: # uniform prior for model parameters
            posterior_current = x_current_loglik # posterior at the current sample

        if is_sigma2_err_unknown: # normal prior for sigma2_err
            # invgammaprior = np.log(invgamma(a = shape_gamma, scale = scale_gamma).pdf(x_current[0][-1])) # scale = scale parameter, a = shape parameter
            sigma2prior = np.log(norm.pdf(x_current[0][-1], mu[-1], sigma[-1]))
            posterior_current += sigma2prior #invgammaprior # posterior at the current sample

        # Candidate sample generation
        x_new =  transition_model_normal(x_current[0], np.diag(sigma_proposal), lower, upper, nDimensions)
        
        # likelihood for candidate sample
        parmsFit = [*x_new[0][0:nDimensions-3], x_new[0][-1]]
        InterventionMult = x_new[0][nDimensions-3:nDimensions-1]
        x_new_loglik, candidate_residual = compute_likelihood(parmsFit, is_sigma2_err_unknown, sigma2_err_known, tmax, fitData, InterventionMult, 
                                                                Population, InitialInfections, InitialExposedMult, lockdown_begin, lockdown_duration)

        if is_prior_normal: # normal prior for model parameters
            normalprior = np.log(multivariate_normal.pdf(x_new[0][:-1], mu[:-1], np.diag(sigma[:-1])))
            posterior_new = x_new_loglik + normalprior # posterior at the candidate sample
        else: # uniform prior for model parameters
            posterior_new = x_new_loglik # posterior at the candidate sample

        if is_sigma2_err_unknown:  # normal prior for sigma2_err
            # invgammaprior = np.log(invgamma(a = shape_gamma, scale = scale_gamma).pdf(x_new[0][-1])) # scale = scale parameter, a = shape parameter
            sigma2prior = np.log(norm.pdf(x_new[0][-1], mu[-1], sigma[-1]))
            posterior_new += sigma2prior # invgammaprior # posterior at the candidate sample
        
        if (acceptance_rule(posterior_current, posterior_new)):
            x_current = x_new.copy()
            if is_sigma2_err_unknown:
                # print(x_current)
                AcceptedUniqueParameters.append(x_new[0])
            else:
                # print(x_current[0][:-1])
                AcceptedUniqueParameters.append(x_new[0][:-1])
            AcceptedResiduals.append(candidate_residual)
            AcceptCounter += 1
            # print('Accepted' + str(AcceptCounter) + ': i=' + str(i))
        else:
            if is_sigma2_err_unknown:
                RejectedParameters.append(x_new[0])
            else:
                RejectedParameters.append(x_new[0][:-1])
            RejectedResiduals.append(candidate_residual)
            RejectCounter += 1
        if is_sigma2_err_unknown:
            PosterierParameters[i,:] = x_current[0]
        else:
            PosterierParameters[i,:] = x_current[0][:-1]

    # use np.loadtxt to evaluate and plot it
    if is_sigma2_err_unknown:
        PrintLabel = SimLabel + '_sigma2err_mu_' + str(mu[-1]) + '_sigma_' + str(sigma[-1]) + '_SigmaPrFact_' + str(proposal_variance_factor) + '_' + countyName
        np.savetxt(StorageFolder + '/AcceptedUniqueParameters_' + SimLabel + '_sigma2err_mu_' + str(mu[-1]) + '_sigma_' + str(sigma[-1]) + \
                   '_SigmaPrFact_' + str(proposal_variance_factor) + '_' + countyName +'.txt', AcceptedUniqueParameters)
    else:
        PrintLabel =  SimLabel + '_sigma2err_' + str(sigma2_err_known) + '_SigmaPrFact_' + str(proposal_variance_factor) + '_' + countyName
        np.savetxt(StorageFolder + '/AcceptedUniqueParameters_' + SimLabel + '_sigma2err_' + str(sigma2_err_known) + '_SigmaPrFact_' + \
                                            str(proposal_variance_factor) + '_' + countyName + '.txt', AcceptedUniqueParameters)

    # use np.loadtxt to evaluate and plot it
    if is_sigma2_err_unknown:
        np.savetxt(StorageFolder + '/AcceptedResiduals_' + SimLabel + '_sigma2err_mu_' + str(mu[-1]) + '_sigma_' + str(sigma[-1]) + \
                   '_SigmaPrFact_' + str(proposal_variance_factor) + '_' + countyName + '.txt', AcceptedResiduals)
    else:
        np.savetxt(StorageFolder + '/AcceptedResiduals_' + SimLabel + '_sigma2err_' + str(sigma2_err_known) + '_SigmaPrFact_' + \
                   str(proposal_variance_factor) + '_' + countyName + '.txt', AcceptedResiduals)

    # use np.loadtxt to evaluate and plot it
    if is_sigma2_err_unknown:
        np.savetxt(StorageFolder + '/RejectedParameters_' + SimLabel + '_sigma2err_mu_' + str(mu[-1]) + '_sigma_' + str(sigma[-1]) + \
                   '_SigmaPrFact_' + str(proposal_variance_factor) + '_' + countyName + '.txt', RejectedParameters)
    else:
        np.savetxt(StorageFolder + '/RejectedParameters_' + SimLabel + '_sigma2err_' + str(sigma2_err_known) + '_SigmaPrFact_' + \
                  str(proposal_variance_factor) + '_' + countyName + '.txt', RejectedParameters)

    # use np.loadtxt to evaluate and plot it
    if is_sigma2_err_unknown:
        np.savetxt(StorageFolder + '/RejectedResiduals_' + SimLabel + '_sigma2err_mu_' + str(mu[-1]) + '_sigma_' + str(sigma[-1]) + \
               '_SigmaPrFact_' + str(proposal_variance_factor) + '_' + countyName + '.txt', RejectedResiduals)
    else:
        np.savetxt(StorageFolder + '/RejectedResiduals_' + SimLabel + '_sigma2err_' + str(sigma2_err_known) + '_SigmaPrFact_' + \
                  str(proposal_variance_factor) + '_' + countyName + '.txt', RejectedResiduals)

    # use np.loadtxt to evaluate and plot it
    if is_sigma2_err_unknown:
        np.savetxt(StorageFolder + '/PosterierParameters_' + SimLabel + '_sigma2err_mu_' + str(mu[-1]) + '_sigma_' + str(sigma[-1]) + \
                                    '_SigmaPrFact_' + str(proposal_variance_factor) + '_' + countyName + '.txt', PosterierParameters)
    else:
        np.savetxt(StorageFolder + '/PosterierParameters_' + SimLabel + '_sigma2err_' + str(sigma2_err_known) + '_SigmaPrFact_' + \
                                            str(proposal_variance_factor) + '_' + countyName + '.txt', PosterierParameters)


    core.RunProjections.RunProjections(AcceptedUniqueParameters, tmax, fitData, Population, InitialInfections, 
        InitialExposedMult, lockdown_begin, lockdown_duration, Dates, PrintLabel, multiprocessingma)
