import numpy as np
import cmath


def LocalEnergy(spins, updatedSpins, NNModel):
	params = NNModel.parameters()
	weights = params[0]
	visBias = params[1]
	hidBias = params[2]
	ELoc = np.array([0. for i in range(len(spins))])
	for k in range(len(spins)):
		ELoc[k] += spins[k] * updatedSpins[k]

		preFact = np.exp(-2*visBias[k]*spins[k])
		multFact = 1
		for i in range(len(hidBias)):
			### Construct theta
			theta = 0.
			for j in range(len(visBias)):
				theta += weights[i][j] * spins[j]
			theta += hidBias[i]
			multFact *= np.cosh(theta - 2 * weights[i][k])/np.cosh(theta)
		multFact *= preFact
		multFact2 = (-1)**((1+spins[k])/2)*complex(0,1)*multFact

		ELoc[k] += multFact
		ELoc[k] += multFact2
		ELoc = np.sum(ELoc)
	return ELoc

def O_Deriv(spins, NNModel):

	params = NNModel.parameters()
	weights = params[0]
	visBias = params[1]
	hidBias = params[2]

	numHid = len(hidBias)
	numVis = len(visbias)


	O_a = np.array([0. for i in range(len(spins))])
	O_b = np.array([0. for i in range(len(hidBias))])
	O_W = np.array([[0. for i in range(len(hidBias))] for j in range(len(spins))])
	for i in range(len(spins)):
		O_a[i] = spins[i]
	for i in range(len(hidBias)):
		theta = 0.
		for j in range(len(visBias)):
			theta += weights[i][j] * spins[j]
		theta += hidBias[i]

		O_b[i] = np.tanh(theta)

		for j in range(len(visBias)):
			O_W[i][j] = spins[j] * np.tanh(theta)


	O_W = np.flatten(O_W)
	StackDev = np.stack((O_W,O_a,O_b))
	StackDev = StackDev.flatten()
	

	return StackDev

'''
def CovMat(O_a, O_b, O_W, ELoc, numHid, numVis):
	nParams = numHid + numVis + numHid*numVis
	
	O_W = np.flatten(O_W)
	StackDev = np.stack((O_W,O_a,O_b))
	Covariance = np.outer(np.conj(StackDev.T), StackDev)

	return Covariance, StackDev'''