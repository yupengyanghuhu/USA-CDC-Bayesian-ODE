import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.stats


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h



def PlotInfectionSubsystemBounds(tmin, CumIActual, CumDActual, INsim, IHsim, Dsim, file, SimLabel):
	obstime = np.linspace(tmin, len(CumIActual), len(CumIActual))
	simtime = np.linspace(tmin, len(INsim[:,0]), len(INsim[:,0]))
	CumIupper = np.zeros(len(INsim[:,0]))
	CumIlower = np.zeros(len(INsim[:,0]))
	CumImean = np.zeros(len(INsim[:,0]))
	Dupper = np.zeros(len(Dsim[:,0]))
	Dlower = np.zeros(len(Dsim[:,0]))
	Dmean = np.zeros(len(Dsim[:,0]))

	for i in range(len(INsim[:,0])):
		Dmean[i], Dlower[i], Dupper[i] = mean_confidence_interval(Dsim[i,:])
		CumImean[i], CumIlower[i], CumIupper[i] = mean_confidence_interval(IHsim[i,:] + INsim[i,:])


	plt.clf()
	plt.subplot(1, 1, 1)
	# plt.title('Infection R0 = ' + str(R0mean))
	plt.title(SimLabel)
	plt.plot(obstime, CumIActual, 'k', linewidth=2, label='Actual Cumulative Infections',color ='g')
	plt.plot(obstime, CumDActual, 'k', linewidth=2, label='Actual Deaths',color ='r')
	plt.plot(simtime, CumImean, 'k', linewidth=2, label='Model Cumulative Infections',color ='k')
	plt.fill_between(simtime, CumIupper, CumIlower, where= CumIupper > CumIlower, color='k', alpha=.25)
	plt.plot(simtime, Dmean, 'k', linewidth=2, label='Model Deaths',color ='b')
	plt.fill_between(simtime, Dupper, Dlower, where= Dupper > Dlower, color='b', alpha=.25)
	plt.xlabel('Days since 2020-01-22')
	plt.ylabel('Number of Individuals')
	plt.legend()
	plt.tight_layout()
	plt.savefig(''+str(file)+'/'+str(SimLabel)+'.png',dpi=300, bbox_inches='tight')
	# plt.show()

	# plt.clf()
	# plt.subplot(1, 1, 1)
	# # plt.title('Infection R0 = ' + str(R0mean))
	# plt.title(SimLabel)
	# plt.plot(obstime, CumIActual, 'k', linewidth=2, label='Actual Cumulative Infections',color ='g')
	# plt.plot(obstime, CumDActual, 'k', linewidth=2, label='Actual Deaths',color ='r')
	# plt.plot(simtime, IHsim[:,-1] + INsim[:,-1], 'k', linewidth=2, label='Model Cumulative Infections',color ='k')
	# plt.plot(simtime, Dsim[:,-1], 'k', linewidth=2, label='Model Deaths',color ='b')
	# plt.xlabel('Days since 2020-01-22')
	# plt.ylabel('Number of Individuals')
	# plt.legend()
	# plt.tight_layout()
	# plt.savefig(''+str(file)+'/BestFit'+str(SimLabel)+'.png',dpi=300, bbox_inches='tight')
	# plt.show()