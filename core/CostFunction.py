import numpy as np

import core.CountryModel

def ThreeStageSSRFunction(x, TotalTime, fitData, scenario, Population, InitialInfections, InitialExposedMult, lockdown_begin, lockdown_duration):

    # tmin = 1
    # tsteps = tmax = lockdown_begin #57# 185
    # tdom = np.linspace(tmin, tmax, tsteps)
    # InterventionMult = 1 
    # Y0, Constants, CompNames = core.CountryModel.SetupCountryModel(x, InterventionMult, Population, InitialInfections, InitialExposedMult)
    # Yout1 = core.CountryModel.RunCountrySim(core.EpiEquations.EpiEquationsExtended, Y0, tdom, Constants)
    # NewInitialConditions = Yout1[tmax-1,:]

    # # Stage 2 (Intervention)
    # tmax = tsteps = lockdown_duration + 1 #30 #1
    # tdom = np.linspace(tmin, tmax, tsteps)
    # InterventionMult = scenario[0] #1 + lockdown_intensity/100 # #.56 #setting to 1 removes intervention to run baseline model
    # Y0, Constants, CompNames = core.CountryModel.SetupCountryModel(x, InterventionMult, Population, InitialInfections, InitialExposedMult)
    # Yout2 = core.CountryModel.RunCountrySim(core.EpiEquations.EpiEquationsExtended, NewInitialConditions, tdom, Constants)
    # NewInitialConditions = Yout2[tmax-1,:]

    # # Stage 3 (Retrun to Baseline)
    # tmax = tsteps = TotalTime + 1 - (lockdown_begin + lockdown_duration) #1
    # tdom = np.linspace(tmin, tmax, tsteps)
    # InterventionMult = scenario[1] #.75#1
    # Y0, Constants, CompNames = core.CountryModel.SetupCountryModel(x, InterventionMult, Population, InitialInfections, InitialExposedMult)
    # Yout3 = core.CountryModel.RunCountrySim(core.EpiEquations.EpiEquationsExtended, NewInitialConditions, tdom, Constants)

    # ALLout = np.vstack([Yout1,Yout2[+1:,:],Yout3[+1:,:]])
    # S, E, C, IN, IH, D, R, CumC, CumIN, CumIH = core.Utils.AssembleOutputExtended(ALLout, 1)

    tmin = 1
    tsteps = tmax = TotalTime  
    tdom = np.linspace(tmin, tmax, tsteps)
    InterventionMult = [1,scenario[0],scenario[1]]
    Y0, Constants, CompNames = core.CountryModel.SetupCountryModel(x, InterventionMult, Population, InitialInfections, InitialExposedMult)
    Yout = core.CountryModel.RunCountrySim(core.EpiEquations.EpiEquationsExtended, Y0, tdom, Constants)

    ALLout = np.vstack([Yout])
    S, E, C, IN, IH, D, R, CumC, CumIN, CumIH = core.Utils.AssembleOutputExtended(ALLout, 1)

    Smat = S.sum(1)
    Emat = E.sum(1)
    Cmat = C.sum(1)
    INmat = IN.sum(1)
    IHmat = IH.sum(1)
    Dmat = D.sum(1)
    Rmat = R.sum(1)
    CumCmat = CumC.sum(1)
    CumINmat = CumIN.sum(1)
    CumIHmat = CumIH.sum(1)

    PredI = CumINmat + CumIHmat
    PredD = Dmat
    predData = np.array([PredI, PredD])
    SR = (fitData - predData)**2
    SSR = SR.sum(0)
    return SSR

