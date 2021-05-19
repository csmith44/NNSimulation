import numpy as np
import torch
from torch import nn
import networkx as nx
from sympy import *

def HilbertBuild(edges):
	return edges

def ratioFunk(NNModel, spinSite):
	params = NNModel.parameters()
	weights = params[0]
	visBias = params[1]
	hidBias = params[2]
	### theta = b_j + sum W_ij * v_j
	preFact = np.exp(-2*visBias*spinVal)
	multFact = 1
	for i in range(len(visBias)):
		theta = 0.
		for ii in range(len(visBias)):

		multFact *= np.cosh()
		


def MetropolisHastings(Hilbert, steps):
	''' Hilbert: Graph with the given connections''' 
	for i in range(steps):
		spinSite = randint(0,len(Hilbert)-1)




def SR():

