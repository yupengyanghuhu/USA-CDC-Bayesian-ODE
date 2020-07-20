import numpy as np

def CalculateR0(beta1, beta2, theta, gamma1, gamma2, gamma3 = None, hosrate = None, delta = None, model ="Base"):
	if model == "Base":
		R0 = beta1*(1-theta)/gamma1 + beta2*theta/gamma2
	elif model == "BaseDeath":
		R0 = beta1*(1-theta)/gamma1 + beta2*theta/(gamma2 + delta)
	elif model == "Extended" or model == "BaseHospitalization":
		R0 = -beta1*(theta*gamma2*gamma3 - gamma2*gamma3)/(gamma1*gamma2*gamma3) - beta2*(hosrate*theta*gamma1*gamma3 - theta*gamma1*gamma3)/(gamma1*gamma2*gamma3) + beta2*hosrate*theta/gamma3
	return R0

