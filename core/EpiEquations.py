import numpy as np
import core.Utils

########### S-E-C-IN-IH-R-D
def EpiEquationsExtended(Yin, t, intRed, alpha, beta, mu, gamma1, gamma2, gamma3, hosrate, theta, delta):

    # unpack state variables
    (S, E, C, I_N, I_H, D, R, CumC, CumIN, CumIH) = core.Utils.StateVecPartExtended(Yin, 1)

    if t <= 30:
        beta = beta * intRed[0]
    elif t > 30 and t <= 75:
        beta = beta * intRed[1]
    else:
        beta = beta * intRed[2]

    N = sum(S + E + C + I_N + I_H + D + R)
    S_dot = -S*C*alpha*beta/N - S*(I_N + I_H)*beta/N
    E_dot = S*C*alpha*beta/N + S*(I_N + I_H)*beta/N - E*mu
    C_dot = E*mu*(1-theta) - C*gamma1
    I_N_dot = E*mu*(1-hosrate)*theta - I_N*gamma2
    I_H_dot = E*mu*hosrate*theta - I_H*gamma3 - I_H*delta
    D_dot = I_H*delta
    R_dot = C*gamma1 + I_N*gamma2 + I_H*gamma3
    CumC_dot = E*mu*(1-theta)
    CumIN_dot = E*mu*(1-hosrate)*theta
    CumIH_dot = E*mu*hosrate*theta
    
    dYout = np.hstack([S_dot, E_dot, C_dot, I_N_dot, I_H_dot, D_dot, R_dot, CumC_dot, CumIN_dot, CumIH_dot])

    return dYout
