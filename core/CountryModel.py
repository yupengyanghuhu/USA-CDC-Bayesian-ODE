##################################################################################
##################################################################################
# Country Simulation (Python 3.7)
##################################################################################
##################################################################################
from scipy.integrate import odeint
from timeit import default_timer as timer
from datetime import timedelta
import numpy as np
from random import randint

import core.EpiEquations
import core.ParameterSet

def SetupCountryModel(Parms, intRed, Population, initInfected, initExposedMult):

        ParmSet = core.ParameterSet.ParametersExtended(*Parms, Population, initInfected, initExposedMult)
        Parms = (intRed, *ParmSet.Constants)
        Y0 = np.array(ParmSet.Y0).flatten()
        CompartNames = ParmSet.CompartmentNames
        return Y0, Parms, CompartNames


def RunCountrySim(model, y0, t, parms):
    # Simulate
    start = timer()
    Y = odeint(model, y0, t, parms)

    end = timer()
    # print("Run completed in ", timedelta(seconds=end - start))
    results = np.array(Y)

    return results
