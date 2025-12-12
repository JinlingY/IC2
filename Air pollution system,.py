##
from IC2 import IC2
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

xy_dim=12
z_dim=6
hid_dim=128
epoch=50
data = np.load(f'/.../data/airpollution/air_HK.npy')
Net_ground= torch.tensor([[0, 0, 0, 0, 2],
                               [1, 0, 1, 1, 1],
                               [1, 1, 0, 2, 2],
                               [1, 1, 2, 0, 2],
                               [2, 1, 2, 2, 0]])
Net_confd = torch.tensor([[0, 0, 0, 0, 2],
                               [0, 0, 0, 0, 0],
                               [0, 0, 0, 2, 2],
                               [0, 0, 2, 0, 2],
                               [2, 0, 2, 2, 0]])
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weights1 = torch.tensor([0.35, 0.35, 0.26, 0.28, 0.03, 0.001,0.001])
weights2 = torch.tensor([0.5, 0.5, 0.08, 0.4, 0.4, 0.001,0.001])
out_s, causal_index, Net_causal, causal_index0, Net_causal0, causal_index1, Net_causal1=IC2(data,weights1,weights2, xy_dim,z_dim,hid_dim,embedding_dim=xy_dim,time_delay=1,n_neighbors=10,T=1,num_epochs=epoch,device=device)
print("Net_causal:",Net_causal)
print("Net_ground:",Net_ground)

causal_index1=causal_index.detach().cpu()
#Ground truth
labels = ['CVDs', 'RSP', '$NO_2$', '$SO_2$', '$O_3$']
Net_ground1=torch.where(Net_ground == 2, 0, Net_ground)
plt.figure(figsize=(10,8))
plt.imshow(Net_ground1, aspect='auto', cmap='Reds')  
plt.xticks(ticks=np.arange(5), labels=labels, rotation=45, fontsize=20)
plt.yticks(ticks=np.arange(5), labels=labels, rotation=45, fontsize=20)
cbar = plt.colorbar()  
cbar.set_label('Ground truth', fontsize=25)  
cbar.ax.tick_params(labelsize=20)
for i in range(Net_ground1.shape[0]):
    for j in range(Net_ground1.shape[1]):
        plt.text(j, i, f'{Net_ground1[i, j]:.0f}', ha='center', va='center', fontsize=30, color='black')
plt.title('KLD') 
plt.show()

#Causal index
plt.figure(figsize=(10,8))
plt.imshow(causal_index1, aspect='auto', cmap='Reds') 
plt.xticks(ticks=np.arange(5), labels=labels, rotation=45, fontsize=20)
plt.yticks(ticks=np.arange(5), labels=labels, rotation=45, fontsize=20)
cbar = plt.colorbar()  
cbar.set_label('causal strength', fontsize=25) 
cbar.ax.tick_params(labelsize=20)  
for i in range(causal_index1.shape[0]):
    for j in range(causal_index1.shape[1]):
        plt.text(j, i, f'{causal_index1[i, j]:.2f}', ha='center', va='center', fontsize=30, color='black')
plt.title('ICIC') 
plt.show()
