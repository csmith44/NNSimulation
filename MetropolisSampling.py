import numpy as np
import torch
from torch import nn
import networkx as nx
import StochasticReconfiguration as SR
from copy import deepcopy

def HilbertBuild(edges):
	return edges

def ratioFunk(NNModel, spins, spinSite):
	params = NNModel.parameters()
	weights = params[0]
	visBias = params[1]
	hidBias = params[2]

	### theta = b_j + sum W_ij * v_j
	preFact = np.exp(-2*visBias[spinSite]*spins[spinSite])
	multFact = 1
	for i in range(len(hidBias)):
		theta = 0.
		for j in range(len(visBias)):
			theta += weights[i][j] * spins[j]
		theta += hidBias[i]
		multFact *= np.cosh(theta - 2 * weights[i][spinSite])/np.cosh(theta)
	r = random.uniform(0,1)
	if multFact**2 < r:
		if spins[spinSite] == 1:
			spins[spinSite] = -1
		else:
			spins[spinSite] = 1
	return spins

def MetropolisHastings(steps, sampling, NNModel, spins, Hilbert=None):
	''' Hilbert: Graph with the given connections'''

	params = NNModel.parameters()
	weights = params[0]
	visBias = params[1]
	hidBias = params[2]
	totParam = len(weights) + len(visBias) + len(hidBias)

	OFull = np.array([])
	O = np.array([0. for i in range(totParam)])
	EFull = np.array([])
	EAvg = 0.
	count = 0

	for i in range(steps):
		spinSite = randint(0,len(Hilbert)-1)
		updatedSpins = ratioFunk(NNModel, spins, spinSite)

		S_oo = []
		S_o = np.array([0. for i in range(totParam)])
		F_Eo = []
		F_E = []

		if i > sampling:
			count += 1

			O += SR.O_Deriv(spins,NNModel)
			OFull.append(O)

			ELoc = SR.LocalEnergy(spins,updatedSpins)
			EAvg += ELoc

			EFull.append(ELoc)

	OAvg = O/count
	EAvg = EAvg/count
	return OFull, OAvg, EAvg, EFull, spins