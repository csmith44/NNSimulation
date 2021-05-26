import numpy as np

def recenter(OAvg,OFull,EAvg,EFull,Ns):
	xCenter = np.array([])
	eCenter = np.array([])
	for i in range(Ns):
		xCenter.append(1/np.sqrt(Ns-1)*(OFull[i]-OAvg))
		eCenter.append(1/np.sqrt(Ns-1)*(EFull[i]-EAvg))
	return xCenter, eCenter

def ForceCov(xCenter, eCenter):
	F = np.conj(xCenter)*eCenter
	S = np.conj(xCenter)*xCenter.T
	return S, F