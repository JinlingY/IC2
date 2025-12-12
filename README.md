## IC2 (Interventional Dynamical Causality under Latent Confounders)
IC2 decipher interventional dynamical causality based solely on non-interventional data even under latent confounders. IC2 is theoretically grounded in dual orthogonal decomposition theorem in the delay embedding space, and is computationally implemented with the constructed interventional data from observed non-interventional data by deep neural networks.

### Examples
#### 3-variable coupled system
##### 1 import packages
```python
import numpy as np
import torch
from utils import logistic_3_system2
import matplotlib.pyplot as plt
import os
from IC2 import IC2 # import our package
```

##### 2 simlate 3-variable logistic system
```python
noise=0.001# noise strength
beta = 0.35# causal/coupled strength
num_steps = 5000# length of time series
#Parameters about model
xy_dim=12 #embedding dimension of the time series
z_dim=6 #dimension of latent variables
hid_dim=128 #dimension of hidden layyers
#Weight
weights1 = torch.tensor([0.35, 0.35, 0.10, 0.09, 0.28, 0.001,0.001])
weights2 = torch.tensor([0.35, 0.35, 0.22, 0.09, 0.25, 0.001,0.001])

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
```
##### 3 calculate the causal strength by IC2 algorithms
```python
# show the ground truth of the system
#Ground truth
#1 represents the direct causality from node i to node j; 
#0 represents that there no causality between two nodes.
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
```

```python
# API for algorithms 
#Users can fine-tune the weight according to data.
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

#Causality inference by IC2
out_s1, causal_index1, Net_causal1, causal_index10, Net_causal10, causal_index11, Net_causal11=IC2(data1,Net_ground1,weights1,weights2,xy_dim,z_dim,hid_dim,embedding_dim=xy_dim,time_delay=1,n_neighbors=10,T=1,num_epochs=50,device=device)
print("Case1 over")

out_s2, causal_index2, Net_causal2, causal_index20, Net_causal20, causal_index21, Net_causal21=IC2(data2,Net_ground2,weights1,weights2,xy_dim,z_dim,hid_dim,embedding_dim=xy_dim,time_delay=1,n_neighbors=10,T=1,num_epochs=50,device=device)
print("Case2 over")

out_s3, causal_index3, Net_causal3, causal_index30, Net_causal30, causal_index31, Net_causal31=IC2(data3,Net_ground3,weights1,weights2,xy_dim,z_dim,hid_dim,embedding_dim=xy_dim,time_delay=1,n_neighbors=10,T=1,num_epochs=50,device=device)
print("Case3 over")

out_s4, causal_index4, Net_causal4, causal_index40, Net_causal40, causal_index41, Net_causal41=IC2(data4,Net_ground4,weights1,weights2,xy_dim,z_dim,hid_dim,embedding_dim=xy_dim,time_delay=1,n_neighbors=10,T=1,num_epochs=50,device=device)
print("Case4 over")

out_s5, causal_index5, Net_causal5, causal_index50, Net_causal50, causal_index51, Net_causal51=IC2(data5,Net_ground5,weights1,weights2,xy_dim,z_dim,hid_dim,embedding_dim=xy_dim,time_delay=1,n_neighbors=10,T=1,num_epochs=50,device=device)
print("Case5 over")

out_s6, causal_index6, Net_causal6, causal_index60, Net_causal60, causal_index61, Net_causal61=IC2(data6,Net_ground6,weights1,weights2,xy_dim,z_dim,hid_dim,embedding_dim=xy_dim,time_delay=1,n_neighbors=10,T=1,num_epochs=50,device=device)
print("Case6 over")
```
##### 4 Result
```python
print("case1(x->y): ",causal_index1[0,1])
print("case2(non causal): ",causal_index2[0,1])
print("case3(x<-z->y, latent confounder): ",causal_index3[0,1])
print("case4(y->x,inverse causality): ",causal_index4[0,1])
print("case5(x->z->y,indirect causality): ",causal_index5[0,1])
print("case6(x->z,y->z,fan in): ",causal_index6[0,1])
```
### License
MIT License