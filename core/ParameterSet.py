import pandas as pd
import numpy as np

import core.ProcessData

########### S-E-C-IN-IH-R-D
class ParametersExtended:
    def __init__(self, alpha, beta, mu, gamma1, gamma2, gamma3, hosrate, theta, delta, Population, initInfected, initExpMult):
        
        S0 = Population - (initExpMult + 1) * initInfected
        E0 = initExpMult*initInfected
        C0 = 0
        IN0 = 0
        IH0 = initInfected
        R0 = 0
        D0 = 0
        CumC0 = 0
        CumIN0 = 0
        CumIH0 = 0

        self.Y0 = (S0, E0, C0, IN0, IH0, D0, R0, CumC0, CumIN0, CumIH0)
        self.Constants = (alpha, beta, mu, gamma1, gamma2, gamma3, hosrate, theta, delta )
        self.CompartmentNames = ['S','E','C','I_N','I_H','D','R','CumC','CumIN','CumIH']
