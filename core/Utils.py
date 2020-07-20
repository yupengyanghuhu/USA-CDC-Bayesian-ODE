# Partition function
import numpy as np
from numba import jit

@jit(nopython=True)


########### S-E-C-IN-IH-R-D
def StateVecPartExtended(StateVec, numCohorts):
    S = StateVec[0: 1 * numCohorts]
    E = StateVec[1 * numCohorts: 2 * numCohorts]
    C = StateVec[2 * numCohorts: 3 * numCohorts]
    IN = StateVec[3 * numCohorts: 4 * numCohorts]
    IH = StateVec[4 * numCohorts: 5 * numCohorts]
    D = StateVec[5 * numCohorts: 6 * numCohorts]
    R = StateVec[6 * numCohorts: 7 * numCohorts]
    CumC = StateVec[7 * numCohorts: 8 * numCohorts]
    CumIN = StateVec[8 * numCohorts: 9 * numCohorts]
    CumIH = StateVec[9 * numCohorts: 10 * numCohorts]

    return S, E, C, IN, IH, D, R, CumC, CumIN, CumIH

def AssembleOutputExtended(Results, numCohorts):
    trng = range(Results.shape[0])
    hrng = range(numCohorts)

    Smat = np.array([[0 for j in hrng] for i in trng]).astype(np.float)
    Emat = np.array([[0 for j in hrng] for i in trng]).astype(np.float)
    Cmat = np.array([[0 for j in hrng] for i in trng]).astype(np.float)
    INmat = np.array([[0 for j in hrng] for i in trng]).astype(np.float)
    IHmat = np.array([[0 for j in hrng] for i in trng]).astype(np.float)
    Dmat = np.array([[0 for j in hrng] for i in trng]).astype(np.float)
    Rmat = np.array([[0 for j in hrng] for i in trng]).astype(np.float)
    CumCmat = np.array([[0 for j in hrng] for i in trng]).astype(np.float)
    CumINmat = np.array([[0 for j in hrng] for i in trng]).astype(np.float)
    CumIHmat = np.array([[0 for j in hrng] for i in trng]).astype(np.float)


    for i in trng:
        (Smat[i, :], Emat[i, :], Cmat[i, :], INmat[i, :], IHmat[i, :], Dmat[i, :], Rmat[i, :], CumCmat[i,:], CumINmat[i, :], CumIHmat[i, :]) = StateVecPartExtended(Results[i,:], numCohorts)

    return Smat, Emat, Cmat, INmat, IHmat, Dmat, Rmat, CumCmat, CumINmat, CumIHmat

########### S-E-C-IN-IH-R
def StateVecPartBaseHosp(StateVec, numCohorts):
    S = StateVec[0: 1 * numCohorts]
    E = StateVec[1 * numCohorts: 2 * numCohorts]
    C = StateVec[2 * numCohorts: 3 * numCohorts]
    IN = StateVec[3 * numCohorts: 4 * numCohorts]
    IH = StateVec[4 * numCohorts: 5 * numCohorts]
    R = StateVec[5 * numCohorts: 6 * numCohorts]
    CumInf = StateVec[6 * numCohorts: 7 * numCohorts]

    return S, E, C, IN, IH, R, CumInf

def AssembleOutputBaseHosp(Results, numCohorts):
    trng = range(Results.shape[0])
    hrng = range(numCohorts)

    Smat = np.array([[0 for j in hrng] for i in trng]).astype(np.float)
    Emat = np.array([[0 for j in hrng] for i in trng]).astype(np.float)
    Cmat = np.array([[0 for j in hrng] for i in trng]).astype(np.float)
    INmat = np.array([[0 for j in hrng] for i in trng]).astype(np.float)
    IHmat = np.array([[0 for j in hrng] for i in trng]).astype(np.float)
    Rmat = np.array([[0 for j in hrng] for i in trng]).astype(np.float)
    CumInfmat = np.array([[0 for j in hrng] for i in trng]).astype(np.float)

    for i in trng:
        (Smat[i, :], Emat[i, :], Cmat[i, :], INmat[i, :], IHmat[i, :], Rmat[i, :], CumInfmat[i, :]) = StateVecPartBaseHosp(Results[i,:], numCohorts)

    return Smat, Emat, Cmat, INmat, IHmat, Rmat, CumInfmat


########### S-E-C-I-R-D
def StateVecPartBaseDeath(StateVec, numCohorts):
    S = StateVec[0: 1 * numCohorts]
    E = StateVec[1 * numCohorts: 2 * numCohorts]
    C = StateVec[2 * numCohorts: 3 * numCohorts]
    I = StateVec[3 * numCohorts: 4 * numCohorts]
    D = StateVec[4 * numCohorts: 5 * numCohorts]
    R = StateVec[5 * numCohorts: 6 * numCohorts]
    CumInf = StateVec[6 * numCohorts: 7 * numCohorts]

    return S, E, C, I, D, R, CumInf

def AssembleOutputBaseDeath(Results, numCohorts):
    trng = range(Results.shape[0])
    hrng = range(numCohorts)

    Smat = np.array([[0 for j in hrng] for i in trng]).astype(np.float)
    Emat = np.array([[0 for j in hrng] for i in trng]).astype(np.float)
    Cmat = np.array([[0 for j in hrng] for i in trng]).astype(np.float)
    Imat = np.array([[0 for j in hrng] for i in trng]).astype(np.float)
    Dmat = np.array([[0 for j in hrng] for i in trng]).astype(np.float)
    Rmat = np.array([[0 for j in hrng] for i in trng]).astype(np.float)
    CumInfmat = np.array([[0 for j in hrng] for i in trng]).astype(np.float)

    for i in trng:
        (Smat[i, :], Emat[i, :], Cmat[i, :], Imat[i, :], Dmat[i, :], Rmat[i, :], CumInfmat[i, :]) = StateVecPartBaseDeath(Results[i,:], numCohorts)

    return Smat, Emat, Cmat, Imat, Dmat, Rmat, CumInfmat


########### S-E-C-I-R
def StateVecPartBase(StateVec, numCohorts):
    S = StateVec[0: 1 * numCohorts]
    E = StateVec[1 * numCohorts: 2 * numCohorts]
    C = StateVec[2 * numCohorts: 3 * numCohorts]
    I = StateVec[3 * numCohorts: 4 * numCohorts]
    R = StateVec[4 * numCohorts: 5 * numCohorts]
    CumInf = StateVec[5 * numCohorts: 6 * numCohorts]

    return S, E, C, I, R, CumInf

def AssembleOutputBase(Results, numCohorts):
    trng = range(Results.shape[0])
    hrng = range(numCohorts)

    Smat = np.array([[0 for j in hrng] for i in trng]).astype(np.float)
    Emat = np.array([[0 for j in hrng] for i in trng]).astype(np.float)
    Cmat = np.array([[0 for j in hrng] for i in trng]).astype(np.float)
    Imat = np.array([[0 for j in hrng] for i in trng]).astype(np.float)
    Rmat = np.array([[0 for j in hrng] for i in trng]).astype(np.float)
    CumInfmat = np.array([[0 for j in hrng] for i in trng]).astype(np.float)

    for i in trng:
        (Smat[i, :], Emat[i, :], Cmat[i, :], Imat[i, :], Rmat[i, :], CumInfmat[i, :]) = StateVecPartBase(Results[i,:], numCohorts)

    return Smat, Emat, Cmat, Imat, Rmat, CumInfmat
