##
from IC2 import IC2
import numpy as np
import torch
import matplotlib.pyplot as plt

xy_dim=12
z_dim=6
hid_dim=128
epoch=100
data = np.load(f'/.../data/foodchain/food_chain.npy')
Net_ground= torch.tensor([[0, 2, 0, 0],
                                [2, 0, 0, 0],
                                [1, 1, 0, 2],
                                [1, 1, 2, 0]])
Net_confd = torch.tensor([[0, 2, 0, 0],
                                [2, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0]])
weights1 = torch.tensor([0.35, 0.35, 0.3, 0.09, 0.1,0.001,0.001])
weights2 = torch.tensor([0.5, 0.5, 0.25, 0.9, 0.5, 0.001,0.001])
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

out_s, causal_index, Net_causal, causal_index0, Net_causal0, causal_index1, Net_causal1=IC2(data,weights1,weights2, xy_dim,z_dim,hid_dim,embedding_dim=xy_dim,time_delay=1,n_neighbors=10,T=3,num_epochs=epoch,device=device)
print("Net_ground:",Net_ground)
print("Net_causal:",Net_causal)
causal_index1=causal_index.detach().cpu()

#Ground truth
labels = ['Calanoid..', 'Rotifers', 'Nano..', 'Pico..']
Net_ground1=torch.where(Net_ground == 2, 0, Net_ground)
plt.figure(figsize=(10,8))
plt.imshow(Net_ground1, aspect='auto', cmap='Reds')  
plt.xticks(ticks=np.arange(4), labels=labels, rotation=45, fontsize=20)
plt.yticks(ticks=np.arange(4), labels=labels, rotation=45, fontsize=20)
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
plt.xticks(ticks=np.arange(4), labels=labels, rotation=45, fontsize=20)
plt.yticks(ticks=np.arange(4), labels=labels, rotation=45, fontsize=20)
cbar = plt.colorbar() 
cbar.set_label('causal strength', fontsize=25) 
cbar.ax.tick_params(labelsize=20)  
for i in range(causal_index1.shape[0]):
    for j in range(causal_index1.shape[1]):
        plt.text(j, i, f'{causal_index1[i, j]:.2f}', ha='center', va='center', fontsize=30, color='black')
plt.show()
import numpy as np