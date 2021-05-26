import numpy as np
import torch
from torch import nn
import MetropolisSampling as ms
import MinResQLP as mqlp
import StochasticReconfiguration
import recenter as rc


class RBM(nn.Module):
   def __init__(self,
               n_vis=5,
               n_hid=100,
               k=5):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_hid,n_vis)*1e-2)
        self.v_bias = nn.Parameter(torch.zeros(n_vis))
        self.h_bias = nn.Parameter(torch.zeros(n_hid))
        self.k = k
    
   def sample_from_p(self,p):
       return F.relu(torch.sign(p - Variable(torch.rand(p.size()))))
    
   def v_to_h(self,v):
        p_h = F.sigmoid(F.linear(v,self.W,self.h_bias))
        sample_h = self.sample_from_p(p_h)
        return p_h,sample_h
    
   def h_to_v(self,h):
        p_v = F.sigmoid(F.linear(h,self.W.t(),self.v_bias))
        sample_v = self.sample_from_p(p_v)
        return p_v,sample_v
        
   def forward(self,v):
        pre_h1,h1 = self.v_to_h(v)
        
        h_ = h1
        for _ in range(self.k):
            pre_v_,v_ = self.h_to_v(h_)
            pre_h_,h_ = self.v_to_h(v_)
        
        return v,v_
    
   def free_energy(self,v):
        vbias_term = v.mv(self.v_bias)
        wx_b = F.linear(v,self.W,self.h_bias)
        hidden_term = wx_b.exp().add(1).log().sum(1)
        return (-hidden_term - vbias_term).mean()

Test = RBM()

params = list(Test.parameters())


spins = [-1,1,-1,1,1]
for i in range(1000):
  count = 0
  gamma = 0.01

  ### SamplingData - OFull, OAvg, EFull, EAvg, spins
  Ns = 500
  OFull,OAvg,EFull,EAvg,spins = ms.MetropolisHastings(steps=1000, sampling=Ns,NNModel=Test,spins=spins)
  xCenter, eCenter = rc.recenter(OAvg,OFull,EAvg,EFull,Ns)
  S,F = rc.ForceCov(xCenter,eCenter)
  Nu = mqlp.minresQLP(S,F)
  for param in params:
    param -= gamma*Nu[count]
    count += 1