import torch
import numpy as np
from IC2.utils import generate_embedd_data,generate_KNN_data,CausalNet_judge_CIC,move_to_device,CausalNet_judge_IntCIC
from IC2.model import DualVAE,train_dualvae,evaluate_CIC,TripleVAE,train_triplevae,evaluate_IntCIC


def IC2(data,weights1,weights2,xy_dim,z_dim,hid_dim,embedding_dim,time_delay,n_neighbors,T,num_epochs,device):
    torch.manual_seed(1991)
    np.random.seed(1991)
    N=data.shape[1]
    X_0_T, X_1_T, X_0_V, X_1_V,X_0,X_1=generate_embedd_data(data,embedding_dim, time_delay, n_neighbors)
    X_0_T = move_to_device(X_0_T, device)
    X_1_T = move_to_device(X_1_T, device)
    X_0_V = move_to_device(X_0_V, device)
    X_1_V = move_to_device(X_1_V, device)
    X_0 = move_to_device(X_0, device)
    X_1 = move_to_device(X_1, device)    
    print("data generating over")
    MSE_s00=torch.zeros(N,N, device=device);MSE_s11=torch.zeros(N,N, device=device)
    MSE_zx00=torch.zeros(N,N, device=device);MSE_zx11=torch.zeros(N,N, device=device)
    MSE_s0=torch.zeros(N,N, device=device);MSE_s1=torch.zeros(N,N, device=device)
    MSE_zx0=torch.zeros(N,N, device=device);MSE_zx1=torch.zeros(N,N, device=device) 
    av_MSE_s0=torch.zeros(N,N, device=device);av_MSE_s1=torch.zeros(N,N, device=device)
    av_MSE_zx0=torch.zeros(N,N, device=device);av_MSE_zx1=torch.zeros(N,N, device=device)
    causal_index0=torch.zeros(N,N, device=device);causal_index1=torch.zeros(N,N, device=device);causal_index=torch.zeros(N,N, device=device)
    Net_causal0=torch.zeros(N,N, device=device);Net_causal1=torch.zeros(N,N, device=device);Net_causal=torch.zeros(N,N, device=device)
    out_s0 = {};out_s1 = {}
    
    L=data.shape[0]-embedding_dim
    Tr=round((L)*0.7)
    Vr=L-Tr
    inter_E=torch.zeros(N,N,Vr)
    for i in range(N):
        for j in range(N):
            if i != j: 
                count0 = 0       
                for t in range(T):
                    model0 = DualVAE(x_dim=xy_dim, y_dim=xy_dim, zx_dim=z_dim, zy_dim=z_dim, s_dim=z_dim, hidden_dims=[hid_dim, hid_dim, hid_dim]).to(device)                
                    model_T0 = train_dualvae(model0, weights1, X_0_T[f"X{i}"],X_1_T[f"X{j}"],device,num_epochs, batch_size=128, learning_rate=1e-3).to(device)
                    out00,MSE_s00[i,j],MSE_zx00[i,j] = evaluate_CIC(model_T0,weights1,X_0_V[f"X{i}"],X_1_V[f"X{j}"],device)

                    MSE_s0[i,j] += MSE_s00[i,j].item()
                    MSE_zx0[i,j] += MSE_zx00[i,j].item()
                    #out_s0[f"s{i},s{j}"] = out00.sample["s_x"]
                    count0 += 1
                av_MSE_s0[i,j] = MSE_s0[i,j] / count0
                av_MSE_zx0[i,j] = MSE_zx0[i,j] / count0
                causal_index0[i,j]=av_MSE_zx0[i,j]
                #print("i",i,"j",j,"Net_ground:", Net_ground[i,j])
                print("i",i,"j",j,"causal_index0:", causal_index0[i,j])
    Net_causal0=CausalNet_judge_CIC(causal_index0,Net_causal0)
    print("Net_causal0:",Net_causal0)
                
                #####
    for i in range(N):
        for j in range(N):
            if i != j: 
                n_datasetX_T, n_datasetY_T, n_datasetX_V, n_datasetY_V=generate_KNN_data(data,X_0[f"X{i}"], X_1[f"X{j}"],embedding_dim, n_neighbors)
                n_datasetX_T= n_datasetX_T.to(device)
                n_datasetY_T= n_datasetY_T.to(device)
                n_datasetX_V= n_datasetX_V.to(device) 
                n_datasetY_V= n_datasetY_V.to(device)
                if Net_causal0[i,j]!=1:  
                    Net_causal[i,j]= Net_causal0[i,j]
                    causal_index[i,j]=causal_index0[i,j]
                else:
                    count1 = 0       
                    for t in range(T):
                        model1 = TripleVAE(dx_dim=xy_dim, dy_dim=xy_dim, dzx_dim=z_dim, dzy_dim=z_dim, ds_dim=z_dim, hidden_dims=[hid_dim, hid_dim, hid_dim],n_neighbors=n_neighbors).to(device)
                        model_T1 = train_triplevae(model1,weights2, n_datasetX_T,n_datasetY_T, device,num_epochs, batch_size=128, learning_rate=1e-3)
                        out11,MSE_s11[i,j],MSE_zx11[i,j] = evaluate_IntCIC(model_T1,weights2,n_datasetX_V,n_datasetY_V, device)

                        MSE_s1[i,j] += MSE_s11[i,j].item()
                        MSE_zx1[i,j] += MSE_zx11[i,j].item()
                        #out_s1[f"s{i},s{j}"] = out11.sample["ds_x"]
                        count1 += 1
                    av_MSE_s1[i,j] = MSE_s1[i,j] / count1
                    av_MSE_zx1[i,j] = MSE_zx1[i,j] / count1

                    causal_index1[i,j]=av_MSE_zx1[i,j] 
                    Net_causal1[i,j]=CausalNet_judge_IntCIC(causal_index1[i,j])
                    if Net_causal1[i,j]== 1 :
                        causal_index[i,j]=causal_index0[i,j]
                        Net_causal[i,j]= Net_causal0[i,j]
                    else:
                        causal_index[i,j]=causal_index1[i,j]
                        Net_causal[i,j]= Net_causal1[i,j]
                
                #print("i",i,"j",j,"Net_ground:", Net_ground[i,j])
                print("i",i,"j",j,"causal_index1:", causal_index1[i,j])
                #print("Net_causal1:", Net_causal1[i,j])
    print("Net_causal:",Net_causal)
    return out_s0, causal_index, Net_causal, causal_index0, Net_causal0, causal_index1, Net_causal1#, inter_E



