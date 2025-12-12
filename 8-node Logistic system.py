##
from IC2 import IC2
import numpy as np
import torch
from IC2.utils import logistic_8_system,Confounder
import matplotlib.pyplot as plt

noise=0.001
beta = 0.35
num_steps = 5000#500;1000;5000;10000 
data = logistic_8_system(noise=noise, beta=beta, num_steps=num_steps)
xy_dim=12
z_dim=6
hid_dim=128

orig_net= torch.tensor([[0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 1, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 1],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0]])
Net_ground,Net_confd= Confounder(orig_net)
weights1 = torch.tensor([0.35, 0.35, 0.15, 0.06, 0.27, 0.001,0.001])
weights2 = torch.tensor([0.35, 0.35, 0.32, 0.06, 0.35, 0.001,0.001])
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
out_s, causal_index, Net_causal, causal_index0, Net_causal0, causal_index1, Net_causal1=IC2(data,weights1,weights2,xy_dim,z_dim,hid_dim,embedding_dim=xy_dim,time_delay=1,n_neighbors=10,T=1,num_epochs=50,device=device)
print("Net_ground:",orig_net)
print("causal_index:",causal_index)

causal_index1=causal_index.detach().cpu()

#Ground truth
Net_ground1=torch.where(Net_ground == 2, 0, Net_ground)
labels = ['node1', 'node2', 'node3', 'node4', 'node5', 'node6', 'node7', 'node8']
plt.figure(figsize=(10,8))
plt.imshow(Net_ground1, aspect='auto', cmap='Reds') 
plt.xticks(ticks=np.arange(8), labels=labels, rotation=45, fontsize=20)
plt.yticks(ticks=np.arange(8), labels=labels, rotation=45, fontsize=20)
cbar = plt.colorbar() 
cbar.set_label('Ground truth', fontsize=25) 
cbar.ax.tick_params(labelsize=20)
for i in range(Net_ground1.shape[0]):
    for j in range(Net_ground1.shape[1]):
        plt.text(j, i, f'{Net_ground1[i, j]:.0f}', ha='center', va='center', fontsize=30, color='black')
plt.show()

#Causal index
plt.figure(figsize=(10,8))
plt.imshow(causal_index1, aspect='auto', cmap='Reds') 
plt.xticks(ticks=np.arange(8), labels=labels, rotation=45, fontsize=15)
plt.yticks(ticks=np.arange(8), labels=labels, rotation=45, fontsize=15)
cbar = plt.colorbar()  
cbar.set_label('causal strength', fontsize=25) 
cbar.ax.tick_params(labelsize=15)  
for i in range(causal_index1.shape[0]):
    for j in range(causal_index1.shape[1]):
        plt.text(j, i, f'{causal_index1[i, j]:.2f}', ha='center', va='center', fontsize=15, color='black')
plt.show()