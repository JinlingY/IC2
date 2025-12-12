import torch
import matplotlib.pyplot as plt
from IC2 import IC2
import os
from IC2.utils import GRN_Dream4_data,Confounder
from IC2.Compare import main_methods,compare_methods

xy_dim=12
z_dim=24
hid_dim=128
epoch=50

GRN_Net, GRN_data=GRN_Dream4_data(n_nold = 10, Net_num=10)
data=GRN_data[f"Net{0}"]      
Net_ground,Net_confd= Confounder(GRN_Net[f"Net{0}"])
weights1 = torch.tensor([0.35, 0.35, 0.30, 0.09,0.17,0.001,0.001])
weights2 = torch.tensor([0.5, 0.5, 0.05, 0.7, 0.8, 0.001,0.001])
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
out_s, causal_index, Net_causal, causal_index0, Net_causal0, causal_index1, Net_causal1=IC2(data,weights1,weights2, xy_dim,z_dim,hid_dim,embedding_dim=xy_dim,time_delay=1,n_neighbors=10,T=1,num_epochs=epoch,device=device)
print("Net_ground:",Net_ground)
print("Net_causal:",Net_causal)
   
res_dir = f'/.../Dream4/net1/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)
Net1=torch.where(Net_causal == 2, 0, Net_causal).detach().cpu()
Net_ground1=torch.where(Net_ground == 2, 0, Net_ground)
causal_index1=causal_index.detach().cpu()
#method_Net_scores=main_methods(data,res_dir,embed_dim=xy_dim,n_neighbor=3,n_excluded=0)
TP, FP, TN, FN, FPR, TPR, Precision0, Precision1, Recall0, Recall1, Accuracy,  roc_auc,methods,CauNet_diff8,Causal_diff8,thrs =compare_methods(res_dir,Net_ground1,causal_index1, Net1,num=40)
print("roc_auc[0]",roc_auc[0])


plt.figure()
plt.plot(FPR[f"X{1}"], TPR[f"X{1}"], color='#1E90FF', lw=2, label='ROC curve (area = %0.3f)(%0.4s) (2-variables)' % (roc_auc[1], methods[1]))
plt.plot(FPR[f"X{2}"], TPR[f"X{2}"], color='#9ACD32', lw=2.2, label='ROC curve (area = %0.3f)(%0.4s) (2-variables)' % (roc_auc[2], methods[2]))
plt.plot(FPR[f"X{3}"], TPR[f"X{3}"], color='#32CD32', lw=2.4, label='ROC curve (area = %0.3f)(%0.4s) (2-variables)' % (roc_auc[3], methods[3]))
plt.plot(FPR[f"X{4}"], TPR[f"X{4}"], color='#2E8B57', lw=2.6, label='ROC curve (area = %0.3f)(%0.4s) (2-variables)' % (roc_auc[4], methods[4]))
plt.plot(FPR[f"X{5}"], TPR[f"X{5}"], color='#B8860B', lw=2.8, label='ROC curve (area = %0.3f)(%0.4s) (2-variables)' % (roc_auc[5], methods[5]))
plt.plot(FPR[f"X{6}"], TPR[f"X{6}"], color='#7A67EE', lw=3, label='ROC curve (area = %0.3f)(%0.4s) (3-variables)' % (roc_auc[6], methods[6]))
plt.plot(FPR[f"X{7}"], TPR[f"X{7}"], color='#4F4F4F', lw=3.2, label='ROC curve (area = %0.3f)(%0.4s) (3-variables)' % (roc_auc[7], methods[7]))
plt.plot(FPR[f"X{8}"], TPR[f"X{8}"], color='#6495ED', lw=2.4, label='ROC curve (area = %0.3f)(%0.6s) (3-variables)' % (roc_auc[8], methods[8]))
plt.plot(FPR[f"X{0}"], TPR[f"X{0}"], color='#EE0000', lw=4, label='ROC curve (area = %0.3f)(%0.4s) (2-variables)' % (roc_auc[0], methods[0]))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([-0.03, 1.0])
plt.ylim([-0.03, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC for GRN_Dream4 network1', fontsize=15)
plt.legend(loc="lower right", fontsize=8)
plt.show()