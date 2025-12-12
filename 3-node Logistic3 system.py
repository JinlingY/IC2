##
from IC2 import IC2
import torch
from IC2.utils import logistic_3_system
import matplotlib.pyplot as plt
import os

noise=0.001
beta =0.35
num_steps = 5000#500;1000;5000;10000 
alpha=1
xy_dim=12
z_dim=6
hid_dim=128
Net_ground1= torch.tensor([[0, 1, 0],
              [0, 0, 0],
              [0, 0, 0]])
Net_ground2= torch.tensor([[0, 0, 0],
              [0, 0, 0],
              [0, 0, 0]])
Net_ground3= torch.tensor([[0, 0, 0],
              [0, 0, 0],
              [1, 1, 0]])
Net_ground4= torch.tensor([[0, 0, 0],
              [1, 0, 0],
              [0, 0, 0]])
Net_ground5= torch.tensor([[0, 0, 1],
              [0, 0, 0],
              [0, 1, 0]])     
Net_ground6= torch.tensor([[0, 0, 1],
              [0, 0, 1],
              [0, 0, 0]])

weights1 = torch.tensor([0.35, 0.35, 0.10, 0.09, 0.28, 0.001,0.001])
weights2 = torch.tensor([0.35, 0.35, 0.22, 0.09, 0.25, 0.001,0.001])

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 1 x->y
data1 = logistic_3_system(noise=noise, betaxy=0, betaxz=0, betayx=beta, betayz=0, betazx=0, betazy=0, alpha=alpha, num_steps=num_steps)
# 2 Non cause 
data2 = logistic_3_system(noise=noise, betaxy=0, betaxz=0, betayx=0, betayz=0, betazx=0, betazy=0, alpha=alpha, num_steps=num_steps)
# 3 x<-z->y, latent confounder
data3 = logistic_3_system(noise=noise, betaxy=0, betaxz=beta, betayx=0, betayz=beta, betazx=0, betazy=0, alpha=alpha, num_steps=num_steps)
# 4 y->x,inverse causality
data4 = logistic_3_system(noise=noise, betaxy=beta, betaxz=0, betayx=0, betayz=0, betazx=0, betazy=0, alpha=alpha, num_steps=num_steps)
# 5 x->z->y,indirect causality
data5 = logistic_3_system(noise=noise, betaxy=0, betaxz=0, betayx=0, betayz=beta, betazx=beta, betazy=0, alpha=alpha, num_steps=num_steps)
# 6 x->z,y->z,fan in
data6 = logistic_3_system(noise=noise, betaxy=0, betaxz=0, betayx=0, betayz=0, betazx=beta, betazy=beta, alpha=alpha, num_steps=num_steps)

out_s1, causal_index1, Net_causal1, causal_index10, Net_causal10, causal_index11, Net_causal11=IC2(data1,weights1,weights2,xy_dim,z_dim,hid_dim,embedding_dim=xy_dim,time_delay=1,n_neighbors=10,T=1,num_epochs=50,device=device)
print("Case1 over")
out_s2, causal_index2, Net_causal2, causal_index20, Net_causal20, causal_index21, Net_causal21=IC2(data2,weights1,weights2,xy_dim,z_dim,hid_dim,embedding_dim=xy_dim,time_delay=1,n_neighbors=10,T=1,num_epochs=50,device=device)
print("Case2 over")
out_s3, causal_index3, Net_causal3, causal_index30, Net_causal30, causal_index31, Net_causal31=IC2(data3,weights1,weights2,xy_dim,z_dim,hid_dim,embedding_dim=xy_dim,time_delay=1,n_neighbors=10,T=1,num_epochs=50,device=device)
print("Case3 over")
out_s4, causal_index4, Net_causal4, causal_index40, Net_causal40, causal_index41, Net_causal41=IC2(data4,weights1,weights2,xy_dim,z_dim,hid_dim,embedding_dim=xy_dim,time_delay=1,n_neighbors=10,T=1,num_epochs=50,device=device)
print("Case4 over")
out_s5, causal_index5, Net_causal5, causal_index50, Net_causal50, causal_index51, Net_causal51=IC2(data5,weights1,weights2,xy_dim,z_dim,hid_dim,embedding_dim=xy_dim,time_delay=1,n_neighbors=10,T=1,num_epochs=50,device=device)
print("Case5 over")
out_s6, causal_index6, Net_causal6, causal_index60, Net_causal60, causal_index61, Net_causal61=IC2(data6,weights1,weights2,xy_dim,z_dim,hid_dim,embedding_dim=xy_dim,time_delay=1,n_neighbors=10,T=1,num_epochs=50,device=device)
print("Case6 over")

print("case1(x->y): ",causal_index1[0,1])
print("case2(non causal): ",causal_index2[0,1])
print("case3(x<-z->y, latent confounder): ",causal_index3[0,1])
print("case4(y->x,inverse causality): ",causal_index4[0,1])
print("case5(x->z->y,indirect causality): ",causal_index5[0,1])
print("case6(x->z,y->z,fan in): ",causal_index6[0,1])

