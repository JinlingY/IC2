
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import random
import pandas as pd



def logistic_3_system(noise, betaxy, betaxz, betayx, betayz, betazx, betazy, alpha, num_steps):
    # 初始化变量Y和X
    rx=alpha *3.7; ry= 3.72; rw=3.78
    Y = np.empty(num_steps)
    X = np.empty(num_steps)
    Z = np.empty(num_steps)
    # 设置初始值
    X[0] = 0.4
    Y[0] = 0.4
    Z[0] = 0.4
    data0=np.zeros((num_steps, 3))
    for j in range(1, num_steps):
        # 计算X和Y的更新值
        X[j] = X[j-1] * (rx - rx * X[j-1]- betaxy * Y[j-1]- betaxz * Z[j-1]) + np.random.normal(0, noise)
        Y[j] = Y[j-1] * (ry - ry * Y[j-1]- betayx * X[j-1]- betayz * Z[j-1]) + np.random.normal(0, noise)
        Z[j] = Z[j-1] * (rw - rw * Z[j-1]- betazx * X[j-1]- betazy * Y[j-1]) + np.random.normal(0, noise)
    
    data0[:,0]=X;data0[:,1]=Y;data0[:,2]=Z
    return data0

def logistic_8_system( noise, beta, num_steps):
    r=np.array([3.9, 3.5, 3.62, 3.75, 3.65, 3.72, 3.57, 3.68])
    data0=np.zeros((num_steps, 8))
    # 初始化变量Y和X
    X1 = np.empty(num_steps);X2 = np.empty(num_steps);X3 = np.empty(num_steps);X4 = np.empty(num_steps)
    X5 = np.empty(num_steps);X6 = np.empty(num_steps);X7 = np.empty(num_steps);X8 = np.empty(num_steps)
    # 设置初始值
    X1[0] = 0.4;X2[0] = 0.4;X3[0] = 0.4;X4[0] = 0.4;X5[0] = 0.4;X6[0] = 0.4;X7[0] = 0.4;X8[0] = 0.4
    for j in range(1, num_steps):
        # 计算X和Y的更新值
        X1[j] = X1[j-1] * (r[0] - r[0] * X1[j-1]) + np.random.normal(0, noise)
        X2[j] = X2[j-1] * (r[1] - r[1] * X2[j-1]) + np.random.normal(0, noise)
        X3[j] = X3[j-1] * (r[2] - r[2] * X3[j-1]- beta * X1[j-1]- beta * X2[j-1]) + np.random.normal(0, noise)
        X4[j] = X4[j-1] * (r[3] - r[3] * X4[j-1]- beta * X2[j-1]) + np.random.normal(0, noise)
        X5[j] = X5[j-1] * (r[4] - r[4] * X5[j-1]- beta * X3[j-1]) + np.random.normal(0, noise)
        X6[j] = X6[j-1] * (r[5] - r[5] * X6[j-1]- beta * X3[j-1]) + np.random.normal(0, noise)
        X7[j] = X7[j-1] * (r[6] - r[6] * X7[j-1]- beta * X6[j-1]) + np.random.normal(0, noise)
        X8[j] = X8[j-1] * (r[7] - r[7] * X8[j-1]- beta * X6[j-1]) + np.random.normal(0, noise)
    data0[:,0]=X1;data0[:,1]=X2;data0[:,2]=X3;data0[:,3]=X4
    data0[:,4]=X5;data0[:,5]=X6;data0[:,6]=X7;data0[:,7]=X8
    return data0

def logistic_8_system_perturbed( noise, beta, num_steps):
    r=np.array([3.9, 3.5, 3.62, 3.75, 3.65, 3.72, 3.57, 3.68])
    N=8
    z=1-np.eye(N, N+1)
    z=z.T
    data0=np.zeros((num_steps, 8))
    Data=np.zeros((N+1,num_steps, 8))
    # 初始化变量Y和X
    X1 = np.empty(num_steps);X2 = np.empty(num_steps);X3 = np.empty(num_steps);X4 = np.empty(num_steps)
    X5 = np.empty(num_steps);X6 = np.empty(num_steps);X7 = np.empty(num_steps);X8 = np.empty(num_steps)
    # 设置初始值
    X1[0] = 0.4;X2[0] = 0.4;X3[0] = 0.4;X4[0] = 0.4;X5[0] = 0.4;X6[0] = 0.4;X7[0] = 0.4;X8[0] = 0.4
    for i in range(N+1):
        for j in range(1, num_steps):
        # 计算X和Y的更新值
            X1[j] = X1[j-1] * (r[0] - r[0] * X1[j-1]) + np.random.normal(0, noise)
            X2[j] = X2[j-1] * (r[1] - r[1] * X2[j-1]) + np.random.normal(0, noise)
            X3[j] = X3[j-1] * (r[2] - r[2] * X3[j-1]- beta *z[i,0] * X1[j-1]- beta *z[i,1]* X2[j-1]) + np.random.normal(0, noise)
            X4[j] = X4[j-1] * (r[3] - r[3] * X4[j-1]- beta *z[i,1] * X2[j-1]) + np.random.normal(0, noise)
            X5[j] = X5[j-1] * (r[4] - r[4] * X5[j-1]- beta *z[i,2] * X3[j-1]) + np.random.normal(0, noise)
            X6[j] = X6[j-1] * (r[5] - r[5] * X6[j-1]- beta *z[i,2] * X3[j-1]) + np.random.normal(0, noise)
            X7[j] = X7[j-1] * (r[6] - r[6] * X7[j-1]- beta *z[i,5] * X6[j-1]) + np.random.normal(0, noise)
            X8[j] = X8[j-1] * (r[7] - r[7] * X8[j-1]- beta *z[i,5] * X6[j-1]) + np.random.normal(0, noise)
        data0[:,0]=X1;data0[:,1]=X2;data0[:,2]=X3;data0[:,3]=X4
        data0[:,4]=X5;data0[:,5]=X6;data0[:,6]=X7;data0[:,7]=X8
        Data[i,:,:]=data0
    return Data

def KLDiv(P, Q):
    P = np.clip(P, 1e-4, None)  # 将 P 中小于 1e-10 的值替换为 1e-10
    Q = np.clip(Q, 1e-4, None)  # 将 Q 中小于 1e-10 的值替换为 1e-10
    if P.shape[1] != Q.shape[1]:
        raise ValueError("The number of columns in P and Q should be the same.")
    
    if not np.all(np.isfinite(P)) or not np.all(np.isfinite(Q)):
        raise ValueError("The inputs contain non-finite values!")

    # Normalizing P and Q
    if Q.shape[0] == 1:
        Q = Q / np.sum(Q)
        P = P / np.sum(P, axis=1, keepdims=True)
        temp = P * np.log(P / Q)
        temp[~np.isfinite(temp)] = 0  # Resolving the case when P(i)==0
        dist = np.sum(temp, axis=1)

    elif Q.shape[0] == P.shape[0]:
        Q = Q / np.sum(Q, axis=1, keepdims=True)
        P = P / np.sum(P, axis=1, keepdims=True)
        temp = P * np.log(P / Q)
        temp[~np.isfinite(temp)] = 0  # Resolving the case when P(i)==0
        dist = np.sum(temp, axis=1)

    return dist

def KLD_perturbed_logistic(L, noise, beta, num_steps):
    N=8
    length = L#20
    start_value = 0.35#0.001
    end_value = 0.36#0.101
    step = (end_value - start_value) / (length)
    beta = np.arange(start_value, end_value, step)
    KLD_num=np.zeros((N, N, L))
    Data=np.zeros((N+1, num_steps, N, L))
    for l in range (L):
        data=logistic_8_system_perturbed(noise, beta[l], num_steps)#m+1,t,m
    # Calculate bin edges
        binedges = np.histogram_bin_edges(data.flatten(), bins='scott')
        nbins = int(np.ceil((len(binedges) - 1) * (N**2 + N)**(-1/3)))
        binedges = np.arange(binedges[0], binedges[-1], (binedges[-1] - binedges[0]) / nbins)
        dist = np.zeros((N+1, len(binedges) - 1, N))
        for i in range(N):# i-th node
            for j in range(N + 1):  #j-th remove
                hist, _ = np.histogram(data[j, :, i], bins=binedges, density=True)
                dist[j, :, i] = hist  # the distribution of i node after removing j node 
        KLD = np.zeros((N, N))            
        for i in range(N):
            KLD[i, :] = KLDiv(dist[i, :, :].T, dist[N, :, :].T)#KLD[i,j]  the difference of p(i) and the p(i) after removing j, i.e.,j->i
        np.fill_diagonal(KLD, 0)   
        print("KLD:",KLD) 
        KLD_num[:,:,l]=KLD[:,:] #KLD_num[i,j,l]:l-th test 2->1
        Data[:,:,:,l]=data[:,:,:]
    return Data, KLD_num

def KLD_perturbed(all_data,data0,Data,M,N,thres):
    KLD_num=np.zeros((N, N))
    # Calculate bin edges
    #binedges0 = np.histogram_bin_edges(data0.flatten(), bins='scott')
    #nbins0 = int(np.ceil((len(binedges0) - 1) * (N**2 + N)**(-1/4)))
    #binedges0 = np.arange(binedges0[0], binedges0[-1], (binedges0[-1] - binedges0[0]) / nbins0)
    #dist0 = np.zeros((len(binedges0) - 1, N))
    #binedges1 = np.histogram_bin_edges(data0.flatten(), bins='scott')
    #nbins1 = int(np.ceil((len(binedges1) - 1) * (N**2 + N)**(-1/4)))
    #binedges1 = np.arange(binedges1[0], binedges1[-1], (binedges1[-1] - binedges1[0]) / nbins1)
    dist = {}#np.zeros((M, len(binedges1) - 1, N))
    KLD = np.zeros((M, N))            
    
    for j in range(M):  #j-th remove
        data_j=Data[f"HAR{j}"]
        binedges1 = np.histogram_bin_edges(data_j.flatten(), bins='scott')
        nbins1 = int(np.ceil((len(binedges1) - 1) * (N**2 + N)**(-1/11)))
        binedges1 = np.arange(binedges1[0], binedges1[-1], (binedges1[-1] - binedges1[0]) / nbins1)
        dist0 = np.zeros((len(binedges1) - 1, N))
        dist1 = np.zeros((len(binedges1) - 1, N))
        for i in range(N):# i-th node
         # the distribution of i node after removing j node 
            hist0, _ = np.histogram(data0[:, i], bins=binedges1, density=True)#True
            dist0[ :, i] = hist0 
            hist, _ = np.histogram(data_j[:, i], bins=binedges1, density=True)
            dist1[:, i] = hist  # the distribution of i node after removing j node 
        #dist[f"f{j}"]  =dist1
        print("j:",j) 
        KLD[j, :] = KLDiv(dist1[ :, :].T, dist0[ :, :].T)#KLD[i,j]  the difference of p(i) and the p(i) after removing j, i.e.,j->i
    #print("KLD:",KLD[i, :]) 
    np.fill_diagonal(KLD, 0)   
    print("KLD:",KLD) 
    Net_ground= np.zeros((M, N))  
    for i in range(M):
        for j in range(N):        
            if KLD[i,j]>=thres:
                Net_ground[i,j]=1
            else:
                Net_ground[i,j]=0   
    return KLD,Net_ground

def KLD_perturbed_Single_cell1(WT_data,KO_data, N):
    # Calculate bin edges
    binedges0 = np.histogram_bin_edges(np.vstack((WT_data, KO_data)).flatten(), bins='scott')
    nbins0 = int(np.ceil((len(binedges0) - 1) * (N**2 + N)**(-1/3)))
    binedges0 = np.arange(binedges0[0], binedges0[-1], (binedges0[-1] - binedges0[0]) / nbins0)
    dist0 = np.zeros((len(binedges0) - 1, N))
    #binedges1 = np.histogram_bin_edges(KO_data.flatten(), bins='scott')
    #nbins1 = int(np.ceil((len(binedges1) - 1) * (N**2 + N)**(-1/3)))
    #binedges1 = np.arange(binedges1[0], binedges1[-1], (binedges1[-1] - binedges1[0]) / nbins1)
    dist1 = np.zeros((len(binedges0) - 1, N))
    for i in range(N):
        hist0, _ = np.histogram(WT_data[:, i], bins=binedges0, density=True)
        dist0[:, i] = hist0  # the distribution of i node after removing j node 
        hist1, _ = np.histogram(KO_data[:, i], bins=binedges0, density=True)
        dist1[:, i] = hist1
    KLD = np.zeros((1, N))              
    KLD[:] = KLDiv(dist0.T, dist1.T)#KLD[i,j]  the difference of p(i) and the p(i) after removing j, i.e.,j->i  
    print("KLD:",KLD) 
        #KLD_num[:,:,l]=KLD[:,:] #KLD_num[i,j,l]:l-th test 2->1
        #Data[:,:,:,l]=data[:,:,:]
    return KLD

def MSE_perturbed_logistic(L, noise, beta, num_steps):
    N=8
    length = L#20
    start_value = 0.35#0.001
    end_value = 0.36#0.101
    step = (end_value - start_value) / (length)
    beta = np.arange(start_value, end_value, step)
    MSE_num=np.zeros((N, N, L))
    Data=np.zeros((N+1, num_steps, N, L))
    for l in range (L):
        data=logistic_8_system_perturbed(noise, beta[l], num_steps)#m+1,t,m
    # Calculate bin edges
        binedges = np.histogram_bin_edges(data.flatten(), bins='scott')
        nbins = int(np.ceil((len(binedges) - 1) * (N**2 + N)**(-1/3)))
        binedges = np.arange(binedges[0], binedges[-1], (binedges[-1] - binedges[0]) / nbins)
        dist = np.zeros((N+1, len(binedges) - 1, N))
        for i in range(N):# i-th node
            for j in range(N + 1):  #j-th remove
                hist, _ = np.histogram(data[j, :, i], bins=binedges, density=True)
                dist[j, :, i] = hist  # the distribution of i node after removing j node 
        MSE = np.zeros((N, N))            
        for i in range(N):
            for j in range(N):
                MSE[i, j] = np.mean((dist[i, :, j] - dist[N, :, j]) ** 2)#KLD[i,j]  the difference of p(i) and the p(i) after removing j, i.e.,j->i
        np.fill_diagonal(MSE, 0)   
        print("KLD:",MSE) 
        MSE_num[:,:,l]=MSE[:,:] #KLD_num[i,j,l]:l-th test 2->1
        Data[:,:,:,l]=data[:,:,:]
    return Data, MSE_num

# 1Lorenz
def lorenz_3(x1, x2, x3,y1, y2, y3,z1, z2, z3,betaxy,betazx,betazy,noise):
    a = 10.0
    b = 28
    c = 8.0 / 3.0
    dx1 = a * (x2 - x1)-betazx*z1+ np.random.normal(0, noise)
    dx2 = x1 * (b - x3) - x2
    dx3 = x1 * x2 - c * x3
    dy1 = a * (y2 - y1)-betaxy*x1-betazy*z1+ np.random.normal(0, noise)
    dy2 = y1 * (b - y3) - y2
    dy3 = y1 * y2 - c * y3
    dz1 = a * (z2 - z1)+ np.random.normal(0, noise)
    dz2 = z1 * (b - z3) - z2
    dz3 = z1 * z2 - c * z3
    return dx1, dx2, dx3, dy1, dy2, dy3, dz1, dz2, dz3

# 2
def simulate_lorenz_3(n_steps, dt, betaxy, betazx, betazy,noise):
    # 初始化状态
    x1 = torch.zeros(n_steps)
    x2 = torch.zeros(n_steps)
    x3 = torch.zeros(n_steps)
    y1 = torch.zeros(n_steps)
    y2 = torch.zeros(n_steps)
    y3 = torch.zeros(n_steps)
    z1 = torch.zeros(n_steps)
    z2 = torch.zeros(n_steps)
    z3 = torch.zeros(n_steps)
    x1[0], x2[0], x3[0] = 1, 1.0, 1
    y1[0], y2[0], y3[0] = 1, 1.0, 1
    z1[0], z2[0], z3[0] = 1, 1.0, 1
    X = torch.zeros((n_steps, 3))
    Y = torch.zeros((n_steps, 3))
    Z = torch.zeros((n_steps, 3))
    # 数值积分
    for i in range(n_steps - 1):
        dx1, dx2, dx3, dy1, dy2, dy3, dz1, dz2, dz3 = lorenz_3(x1[i], x2[i], x3[i],y1[i], y2[i], y3[i],z1[i], z2[i], z3[i],betaxy,betazx,betazy,noise)
        x1[i + 1] = x1[i] + dx1 * dt
        x2[i + 1] = x2[i] + dx2 * dt
        x3[i + 1] = x3[i] + dx3 * dt
        y1[i + 1] = y1[i] + dy1 * dt
        y2[i + 1] = y2[i] + dy2 * dt
        y3[i + 1] = y3[i] + dy3 * dt
        z1[i + 1] = z1[i] + dz1 * dt
        z2[i + 1] = z2[i] + dz2 * dt
        z3[i + 1] = z3[i] + dz3 * dt
    X[:,0]=x1
    X[:,1]=x2
    X[:,2]=x3
    Y[:,0]=y1
    Y[:,1]=y2
    Y[:,2]=y3
    Z[:,0]=z1
    Z[:,1]=z2
    Z[:,2]=z3  
    data= {};data[f"X{0}"]=X;data[f"X{1}"]=Y;data[f"X{2}"]=Y
    return data

def generate_embedd_lorenz(data,n_neighbors):
    L=data[f"X{1}"].shape[0]
    Tr=round(L*0.7)
    X_0_T = {};X_1_T = {};X_0_V = {};X_1_V = {}
    X_0 = {};X_1 = {} 
    for i in range(3):
        x_delay=data[f"X{i}"].numpy()
        x_delay0 = x_delay[:-1, :]; x_delay1 = x_delay[1:, :]#t-1和t
        #x_delay0 = x_delay; x_delay1 = x_delay
        normalizer_x0 = MinMaxNormalize(x_delay0); normalizer_x1 = MinMaxNormalize(x_delay1)
        n_x0 = normalizer_x0.normalize(x_delay0); n_x1 = normalizer_x1.normalize(x_delay1)
        n_x0 =torch.from_numpy(n_x0); n_x1 =torch.from_numpy(n_x1)
        n_x00 =n_x0.unsqueeze(1).repeat(1, n_neighbors, 1);n_x11 =n_x1.unsqueeze(1).repeat(1, n_neighbors, 1)
        X_0_T[f"X{i}"] =n_x00[:Tr,:,:].float(); X_1_T[f"X{i}"] =n_x11[:Tr,:,:].float()
        X_0_V[f"X{i}"] =n_x00[Tr:,:,:].float(); X_1_V[f"X{i}"] =n_x11[Tr:,:,:].float()
        X_0[f"X{i}"] =n_x0.float(); X_1[f"X{i}"] =n_x1.float()
        #X_0[i] = n_x0.float(); X_1[i] = n_x1.float()
    return X_0_T, X_1_T, X_0_V, X_1_V, X_0, X_1

# 2 Lorenz
def RK4(func, X0, sets,dx):
    """
    Runge Kutta 4 solver.
    """
    n_x, dt, n_data, h, c, A, sigma = sets
    func_sets = (n_x, h, c, A, sigma)
    X  = np.zeros([n_data, len(X0)])
    X[0] = X0
    ti = 0
    for i in range(n_data-1):
        k1 = func(X[i], ti, func_sets, dx)
        k2 = func(X[i] + dt/2. * k1, ti + dt/2., func_sets, dx)
        k3 = func(X[i] + dt/2. * k2, ti + dt/2., func_sets, dx)
        k4 = func(X[i] + dt    * k3, ti + dt, func_sets, dx)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
        ti += dt
    return X

def coupled_Lorenz(x, t, sets,dx):
#     Lorenz system is time-invariant, no need to use the input t
    n_x, h, c, A, sigma = sets
    r = np.zeros(x.shape)
    noise = np.random.normal(loc=0, scale=sigma, size=3*n_x)
    for k in range(n_x):
        if k ==0:
            r[k*3+0]= -10*(x[k*3+0]-x[k*3+1] +c*A[k]@(x[1::3]-x[k*3+1])) + noise[0+k*3]+dx
        else:
            r[k*3+0]= -10*(x[k*3+0]-x[k*3+1] +c*A[k]@(x[1::3]-x[k*3+1])) + noise[0+k*3]
        r[k*3+1]= 28*(1+h[k])*x[k*3+0]-x[k*3+1] - x[k*3+0]*x[k*3+2] + noise[1+k*3]
        r[k*3+2]= -8/3*x[k*3+2]+x[k*3+0]*x[k*3+1] + noise[2+k*3]
    return r

def gen_data_norm(n_x, A, n_data, dx, c = 0.3, sigma_dyn = 0, sigma_obs = 0, dt = 0.01):
    h  = 2*(np.random.rand(n_x)-0.5)*0.06
    
    setting = (n_x, dt, n_data, h, c, A, sigma_dyn)
    u0 = np.array([7.432487609628195, 10.02071718705213, 29.62297428638419])
    x0 = u0 + np.random.random((n_x,3))*0.3

#     X = gen_data(setting, x0)
    X = RK4(coupled_Lorenz, x0.flatten(), setting, dx)
    
    #q1 = np.percentile(X, 25, interpolation='midpoint', axis=0)
    #q2 = np.percentile(X, 50, interpolation='midpoint', axis=0)
    #q3 = np.percentile(X, 75, interpolation='midpoint', axis=0)

    #X = (X - q2)/(q3 - q1)
    
    #if sigma_obs != 0:
    #    noise_obs = np.random.normal(loc=0, scale=sigma_obs, size=X.shape)
    #    X += noise_obs
    indices = np.arange(0, n_x * 3, 3)  # 生成索引 [0, 3, 6, ...] 直到 (n-1)*3
    data = X[:, indices]
    #data=X[:,[0,3,6]]
    #data = np.column_stack((
    #X[:, 0:3].reshape(-1),   # 第 1-3 列
    #X[:, 3:6].reshape(-1),   # 第 4-6 列
    #X[:, 6:9].reshape(-1) ))   # 第 7-9 列

    return data

def lorenz_perturbed(n_x, A, n_data, dx, c = 0.3, sigma_dyn = 0, sigma_obs = 0, dt = 0.01):
    
    N=n_x
    z=1-np.eye(N, N+1)
    z=z.T
    data0=np.zeros((n_data, N))
    Data=np.zeros((N+1,n_data, N))

    h  = 2*(np.random.rand(n_x)-0.5)*0.06
    u0 = np.array([7.432487609628195, 10.02071718705213, 29.62297428638419])
    x0 = u0 + np.random.random((n_x,3))*0.3
    for i in range(N+1):
        AA=A.copy()
        if i<N:
            AA[:,i]=0
        else:
            AA=A
        setting = (n_x, dt, n_data, h, c, AA, sigma_dyn)
        X = RK4(coupled_Lorenz, x0.flatten(), setting, dx)
        indices = np.arange(0, n_x * 3, 3)  # 生成索引 [0, 3, 6, ...] 直到 (n-1)*3
        data0 = X[:, indices]
        Data[i,:,:]=data0
    return Data

def KLD_perturbed_lorenz(L, n_x, A, num_steps):
    N=n_x
    length = L#20
    start_value = 0.5#0.001
    end_value = 0.51#0.101
    step = (end_value - start_value) / (length)
    beta = np.arange(start_value, end_value, step)
    KLD_num=np.zeros((N, N, L))
    Data=np.zeros((N+1, num_steps, N, L))
    for l in range (L):
        data=lorenz_perturbed(n_x, A, num_steps, dx=0, c = beta[l], sigma_dyn = 0.001, sigma_obs = 0.001, dt = 0.01)#m+1,t,m
    # Calculate bin edges
        binedges = np.histogram_bin_edges(data.flatten(), bins='scott')
        nbins = int(np.ceil((len(binedges) - 1) * (N**2 + N)**(-1/3)))
        binedges = np.arange(binedges[0], binedges[-1], (binedges[-1] - binedges[0]) / nbins)
        dist = np.zeros((N+1, len(binedges) - 1, N))
        for i in range(N):# i-th node
            for j in range(N + 1):  #j-th remove
                hist, _ = np.histogram(data[j, :, i], bins=binedges, density=True)
                dist[j, :, i] = hist  # the distribution of i node after removing j node 
        KLD = np.zeros((N, N))            
        for i in range(N):
            KLD[i, :] = KLDiv(dist[i, :, :].T, dist[N, :, :].T)#KLD[i,j]  the difference of p(i) and the p(i) after removing j, i.e.,j->i
        np.fill_diagonal(KLD, 0)   
        print("KLD:",KLD) 
        KLD_num[:,:,l]=KLD[:,:] #KLD_num[i,j,l]:l-th test 2->1
        Data[:,:,:,l]=data[:,:,:]
    return Data, KLD_num

class data_CL():
    def __init__(self, args):
        self.noise_sigma = args.noise_sigma
        self.args = args
        self.N = args.N 
        self.n = args.n #network size
        self.T = args.T #total iteration steps
        self.V = args.V #dimension
        self.dt = args.dt
        self.ddt = args.ddt
        self.beta = args.beta
        self.c = args.couple_str
        self.h = 2*(np.random.rand(self.n)-0.5)*0.5
        
        self.net = self.gen_net()
    
    #def gene(self):
    #    Xss = self.gen_data() 
    #    self.sav_data(Xss)
    
    def gen_net(self):
        args = self.args
        if args.direc:
            net = nx.DiGraph() 
        else:
            net = nx.Graph() 
        net.add_nodes_from(np.arange(self.n)) 
        if args.net_nam == 'er': 
            for u, v in nx.erdos_renyi_graph(self.n, 0.6).edges():
                net.add_edge(u,v,weight=random.uniform(0,1))
        elif args.net_nam == 'ba':
            for u, v in nx.barabasi_albert_graph(self.n, 4).edges():
                net.add_edge(u,v,weight=random.uniform(1,1)) 
        elif args.net_nam == 'rg':
            for u, v in nx.random_regular_graph(4,self.n).edges():
                net.add_edge(u,v,weight=random.uniform(1,1))
        elif args.net_nam == 'edges':
            edges = np.array([[0,1],[0,4],[1,0],[1,3],[2,1],[3,4]])
            weights = np.array([self.beta,self.beta,self.beta,self.beta,self.beta,self.beta,self.beta,self.beta,self.beta,self.beta])
            for i in range(edges.shape[0]):
                net.add_edge(edges[i,0],edges[i,1],weight=weights[i])
            print(args.net_nam)
        else:
            edges = pd.read_csv("./dataset/data/CL/edges.csv").values[:,1:]
            weights = pd.read_csv("./dataset/data/CL/weights.csv").values[:,1:]
            for i in range(edges.shape[0]):
                net.add_edge(edges[i,0],edges[i,1],weight=weights[i])
            print(args.net_nam)
            
        for i in range(args.n):
            swei = 0
            for j in net.neighbors(i):
                swei = swei+net.get_edge_data(i,j)['weight'] 
            for j in net.neighbors(i):
                net.edges[i,j]['weight'] = net.edges[i,j]['weight']/swei
        return(net)
    
    # self-dynamics
    def F(self, x, k):
        f = np.zeros(x.shape)
        f[0] = -10*(x[0]-x[1])
        f[1] = 28*(1+self.h[k])*x[0] - x[1] - x[0]*x[2]
        f[2] = -8/3*x[2] + x[0]*x[1]
        return(f)
    
    # coupling-dynamics
    def G(self, x, y):
        g = np.zeros(x.shape)
        g[0] = -10*(y[1]-x[1])
        return(g)
    
    def gen_data(self):
        T, N, n, V, dt, ddt = self.T, self.N, self.n, self.V, self.dt, self.ddt
        net, F, G = self.net, self.F, self.G
        c = self.c
        Xs = np.zeros((T*N,n,V))
        for Ni in range(N):
            # initial condition 
            x = np.zeros((T,n,V))
            x_cur = np.ones((n,V)) + 5*1e-1*np.random.rand(n,V) 
            x[0,:,:] = x_cur
            deln = int(dt/ddt)
            # dynamical equation
            for it in range(T*deln-1):
                for i in range(n):
                    f = F(x_cur[i,:],i)
                    g = 0
                    for j in net.neighbors(i):
                        g += G(x_cur[i,:],x_cur[j,:])*net.get_edge_data(i,j)['weight']
                    dx = (f + c*g)*ddt
                    x_cur[i,:] = x_cur[i,:] + dx
                if (it+1)%deln == 0:
                    x[int((it+1)/deln),:,:] = x_cur
            Xs[Ni*T:(Ni+1)*T,:,:] = x 
        Xss = np.zeros((T*N,n))
        #for Vi in range(V):
        Xss[0*T*N:T*N,:] = Xs[:,:,0]
        noise = self.noise_sigma * np.random.randn(Xss.shape[0],Xss.shape[1])
        Xss += noise
        time_point = np.arange(self.T)*self.dt
        return(Xss,time_point)

class data_Rossler():
    def __init__(self, args):
        self.noise_sigma = args.noise_sigma
        self.args = args
        self.N = args.N 
        self.n = args.n #network size
        self.T = args.T #total iteration steps
        self.V = args.V #dimension
        self.dt = args.dt
        self.ddt = args.ddt
        self.beta = args.beta
        self.c = 0.2
        #self.w = 1.0 + 0.2*(np.random.rand(args.n)-0.5)
        self.w = 1.0 + 0*(np.random.rand(args.n)-0.5)
        
        self.net = self.gen_net()
    
    #def gene(self):
    #    Xss = self.gen_data() 
    #    self.sav_data(Xss)
    
    def gen_net(self):
        args = self.args
        if self.args.direc:
            net = nx.DiGraph() 
        else:
            net = nx.Graph()
        net.add_nodes_from(np.arange(self.n))
        if self.args.net_nam == 'er':
            for u, v in nx.erdos_renyi_graph(self.n, 0.2).edges():
                net.add_edge(u,v,weight=random.uniform(1,1))
        elif self.args.net_nam == 'ba':
            for u, v in nx.barabasi_albert_graph(self.n, 4).edges():
                net.add_edge(u,v,weight=random.uniform(1,1)) 
        elif self.args.net_nam == 'rg':
            for u, v in nx.random_regular_graph(3,self.n).edges():
                net.add_edge(u,v,weight=random.uniform(1,1))
        elif args.net_nam == 'edges':
            edges = np.array([[0,1],[0,4],[1,0],[1,3],[2,1],[3,4]])
            weights = np.array([self.beta,self.beta,self.beta,self.beta,self.beta,self.beta,self.beta,self.beta,self.beta,self.beta])
            for i in range(edges.shape[0]):
                net.add_edge(edges[i,0],edges[i,1],weight=weights[i])
            print(args.net_nam)
        else:
            edges = pd.read_csv("./dataset/data/Rossler/edges.csv").values[:,1:]
            weights = pd.read_csv("./dataset/data/Rossler/weights.csv").values[:,1:]
            for i in range(edges.shape[0]):
                net.add_edge(edges[i,0],edges[i,1],weight=weights[i])
            print(args.net_nam)
            
        for i in range(args.n):
            swei = 0
            for j in net.neighbors(i):
                swei = swei+net.get_edge_data(i,j)['weight'] 
            for j in net.neighbors(i):
                net.edges[i,j]['weight'] = net.edges[i,j]['weight']/swei
        return(net)
    
    # self-dynamics
    def F(self, x, k):
        f = np.zeros(x.shape)
        f[0] = -self.w[k]*x[1] - x[2]
        f[1] = self.w[k]*x[0] + 0.2*x[1]
        f[2] = 0.2 + x[2]*(x[0]-6)
        return(f)
    
    # coupling-dynamics
    def G(self, x, y):
        g = np.zeros(x.shape)
        g[0] = (y[0]-x[0])*self.args.couple_str
        return(g)
    
    def gen_data(self):
        T, N, n, V, dt, ddt = self.T, self.N, self.n, self.V, self.dt, self.ddt
        net, F, G = self.net, self.F, self.G
        c = self.c
        Xs = np.zeros((T*N,n,V))
        for Ni in range(N):
            # initial condition 
            x = np.zeros((T,n,V))
            x_cur = 2*np.ones((n,V)) + 5*1e-2*np.random.rand(n,V) 
            x[0,:,:] = x_cur
            deln = int(dt/ddt)
            # dynamical equation
            for it in range(T*deln-1):
                for i in range(n):
                    f = F(x_cur[i,:],i)
                    g = 0
                    for j in net.neighbors(i):
                        g += G(x_cur[i,:],x_cur[j,:])*net.get_edge_data(i,j)['weight']
                    dx = (f + c*g)*ddt
                    x_cur[i,:] = x_cur[i,:] + dx
                if (it+1)%deln == 0:
                    x[int((it+1)/deln),:,:] = x_cur
            Xs[Ni*T:(Ni+1)*T,:,:] = x    
        Xss = np.zeros((T*N,n))
        #for Vi in range(V):
        Xss[0*T*N:T*N,:] = Xs[:,:,0]
        noise = self.noise_sigma * np.random.randn(Xss.shape[0],Xss.shape[1])
        Xss += noise
        time_point = np.arange(self.T)*self.dt
        return(Xss,time_point)

def simu_henon_9v_1(alpha, n_iter, noise, xs0=None, xs1=None):
    """
    perform the simulation of 9-variable Henon system.

    Parameters
    ----------
    xs0: vector or 1d array
        nine initial x0
    xs1: vector or 1d array
        nine initial x1
    alpha: float
        parameter of henon system
    n_iter: int
        maximal number of iteration
    noise: float
        noise intensity

    Returns
    -------
    sim_array: 2d array
        simulated output array
    """
    n_known = 2
    n_feature = 9

    if xs0 is None:
        xs0 = np.arange(0.1, 1., 0.1)
    x1_0, x2_0, x3_0, x4_0, x5_0, x6_0, x7_0, x8_0, x9_0 = xs0

    if xs1 is None:
        xs1 = np.arange(0.9, 0., -0.1)
    x1_1, x2_1, x3_1, x4_1, x5_1, x6_1, x7_1, x8_1, x9_1 = xs1

    sim_array = np.zeros((n_iter + n_known, n_feature))
    sim_array[0] = xs0
    sim_array[1] = xs1
    for i in range(n_known, n_iter + n_known):
        x1 = 1.4 - x1_1 ** 2 + 0.3 * x1_0 + np.random.normal(0, noise)
        #x9 = 1.4 - x9_1 ** 2 + 0.3 * x9_0 + np.random.normal(0, noise)
        x2 = 1.4 - (0.5 * alpha * (x1_1 ) + (1.0 - alpha) * x2_1) ** 2 + \
             0.3 * x2_0 + np.random.normal(0, noise)
        x3 = 1.4 - (0.5 * alpha * (x2_1 ) + (1.0 - alpha) * x3_1) ** 2 + \
             0.3 * x3_0 + np.random.normal(0, noise)
        x4 = 1.4 - (0.5 * alpha * (x3_1 ) + (1.0 - alpha) * x4_1) ** 2 + \
             0.3 * x4_0 + np.random.normal(0, noise)
        x5 = 1.4 - (0.5 * alpha * (x4_1 ) + (1.0 - alpha) * x5_1) ** 2 + \
             0.3 * x5_0 + np.random.normal(0, noise)
        x6 = 1.4 - (0.5 * alpha * (x5_1 ) + (1.0 - alpha) * x6_1) ** 2 + \
             0.3 * x6_0 + np.random.normal(0, noise)
        x7 = 1.4 - (0.5 * alpha * (x6_1 ) + (1.0 - alpha) * x7_1) ** 2 + \
             0.3 * x7_0 + np.random.normal(0, noise)
        x8 = 1.4 - (0.5 * alpha * (x7_1 ) + (1.0 - alpha) * x8_1) ** 2 + \
             0.3 * x8_0 + np.random.normal(0, noise)
        x9 = 1.4 - (0.5 * alpha * (x8_1 ) + (1.0 - alpha) * x9_1) ** 2 + \
             0.3 * x9_0 + np.random.normal(0, noise)
        #x9 = 1.4 - x9_1 ** 2 + 0.3 * x9_0 + np.random.normal(0, noise)
        sim_array[i] = [x1, x2, x3, x4, x5, x6, x7, x8, x9]
        x1_0, x2_0, x3_0, x4_0, x5_0, x6_0, x7_0, x8_0, x9_0 = \
            x1_1, x2_1, x3_1, x4_1, x5_1, x6_1, x7_1, x8_1, x9_1
        x1_1, x2_1, x3_1, x4_1, x5_1, x6_1, x7_1, x8_1, x9_1 = \
            x1, x2, x3, x4, x5, x6, x7, x8, x9
        
    return sim_array

def edges_to_mat(edges, n_nold=None):
    assert edges.ndim == 2 and edges.shape[1] >= 2
    nolds = np.max(edges) + 1
    if n_nold is None or n_nold < nolds:
        n_nold = nolds
    mat = np.zeros((n_nold, n_nold))
    mat[edges[:, 0], edges[:, 1]] = 1
    return mat

def GRN_Dream4_data(n_nold = 10, Net_num=10):
    GRN_Net = {};GRN_data = {}
    #n_nold = 10
    #Net_num=10
    for j in range(Net_num):
        Net_posi = pd.read_csv(f'/.../data/gene/net{j+1}_truth.tsv', delimiter='\s+', header=None)
        data_posi = np.load(f'/.../data/gene/net{j+1}_expression.npy')
        #Network
        Net_posi.iloc[:, :2] = Net_posi.iloc[:, :2].apply(lambda x: [int(v.split('G')[1]) for v in x])
        edges_truth = Net_posi[[0, 1]].values.astype('int')
        gold_mat = edges_to_mat(edges_truth - 1, n_nold)
        #truths = skip_diag_tri(gold_mat).ravel()
        GRN_Net[f"Net{j}"] =torch.from_numpy(gold_mat)
        #data
        GRN_data[f"Net{j}"]=data_posi
    return GRN_Net, GRN_data

def HAR151_data(Net_num):
    GRN_data = {}
    #n_nold = 10
    #Net_num=10
    for j in range(Net_num):
        HAR_data = np.loadtxt(f'/.../data/GSE270828/import/HAR_{j+1}.csv', delimiter=',')
        #data
        GRN_data[f"HAR{j}"]=HAR_data
    Control_data=np.loadtxt(f'/.../data/GSE270828/import/Control.csv', delimiter=',')
    return GRN_data, Control_data

def Pert4_data(Net_num):
    GRN_data = {}
    #n_nold = 10
    #Net_num=10
    for j in range(Net_num):
        Pert_data = np.loadtxt(f'/.../data/GSE249416/import/Pert_{j+1}.csv', delimiter=',')
        #data
        GRN_data[f"HAR{j}"]=Pert_data
    Control_data=np.loadtxt(f'/.../data/GSE249416/import/Control.csv', delimiter=',')
    return GRN_data, Control_data

def Pert4_data1(Net_num):
    GRN_data = {}
    #n_nold = 10
    #Net_num=10
    for j in range(Net_num):
        Pert_data = np.loadtxt(f'/.../data/GSE249416/import1/Pert_{j+1}.csv', delimiter=',')
        #data
        GRN_data[f"HAR{j}"]=Pert_data
    Control_data=np.loadtxt(f'/.../data/GSE249416/import1/Control.csv', delimiter=',')
    return GRN_data, Control_data

def HAR151_data_1(Net_num,Gene_id):
    GRN_data = {}
    #n_nold = 10
    #Net_num=10
    for j in range(Net_num):
        HAR_data0 = np.loadtxt(f'/.../data/GSE270828/import/HAR_{j+1}.csv', delimiter=',')
        HAR_data=HAR_data0[:,Gene_id.astype(int)-1]
        GRN_data[f"HAR{j}"]=HAR_data
    Control_data0=np.loadtxt(f'/.../data/GSE270828/import/Control.csv', delimiter=',')
    Control_data=Control_data0[:,Gene_id.astype(int)-1]    
    return GRN_data, Control_data

def HAR151_data_2(Net_num,Gene_id,Pert_id):
    GRN_data = {}
    #n_nold = 10
    #Net_num=10
    for j in range(Net_num):
        HAR_data0 = np.loadtxt(f'/.../data/GSE270828/import/HAR_{j+1}.csv', delimiter=',')
        HAR_data=HAR_data0[Pert_id.astype(int)-1,Gene_id.astype(int)-1]
        GRN_data[f"HAR{j}"]=HAR_data
    Control_data0=np.loadtxt(f'/.../data/GSE270828/import/Control.csv', delimiter=',')
    Control_data=Control_data0[Pert_id.astype(int)-1,Gene_id.astype(int)-1]    
    return GRN_data, Control_data

def generate_embedd_KNN_data(data,embedding_dim, time_delay,n_neighbors):
    ##here _0 represents "t", _1 represents "t-1".
    ## _T is the samples for training, _V is the samples for validation.
    X_dict0_T = {};X_dict1_T = {};X_dict0_V = {};X_dict1_V = {}
    L=data.shape[0]-embedding_dim+1-1
    Tr=round(L*0.7)
    Vr=L-Tr
    NN=n_neighbors
    E=embedding_dim
    XNN_0_L = torch.zeros(L, NN, E); XNN_1_L = torch.zeros(L, NN, E)
    XNN_0_T = torch.zeros(Tr, NN, E); XNN_1_T = torch.zeros(Tr, NN, E)
    XNN_0_V = torch.zeros(Vr, NN, E); XNN_1_V = torch.zeros(Vr, NN, E)
    X_0_T = {};X_1_T = {};X_0_V = {};X_1_V = {}
    n_dataset0_T = TensorDataset();n_dataset1_T = TensorDataset()
    n_dataset0_V = TensorDataset();n_dataset1_V = TensorDataset()
    
    for i in range(data.shape[1]):
        x_delay=delay_embedding(data[:,i], embedding_dim, time_delay)
        x_delay0 = x_delay[:-1, :]; x_delay1 = x_delay[1:, :]#t-1和t
        #x_delay0 = x_delay; x_delay1 = x_delay
        normalizer_x0 = MinMaxNormalize(x_delay0); normalizer_x1 = MinMaxNormalize(x_delay1)
        n_x0 = normalizer_x0.normalize(x_delay0); n_x1 = normalizer_x1.normalize(x_delay1)
        n_x0 =torch.from_numpy(n_x0); n_x1 =torch.from_numpy(n_x1)
        n_x00 =n_x0.unsqueeze(1).repeat(1, n_neighbors, 1);n_x11 =n_x1.unsqueeze(1).repeat(1, n_neighbors, 1)
        X_0_T[f"X{i}"] =n_x00[:Tr,:,:].float(); X_1_T[f"X{i}"] =n_x11[:Tr,:,:].float()
        X_0_V[f"X{i}"] =n_x00[Tr:,:,:].float(); X_1_V[f"X{i}"] =n_x11[Tr:,:,:].float()
        #X_0[i] = n_x0.float(); X_1[i] = n_x1.float()
    
        for j in range(n_x0.shape[0]): 
            X_0_KNN = torch.cat([n_x0[:j], n_x0[j+1:]], dim=0);X_1_KNN = torch.cat([n_x1[:j], n_x1[j+1:]], dim=0)   
            XNN_0=KNN(n_x0[j],X_0_KNN,n_neighbors)
            XNN_1=KNN(n_x1[j],X_1_KNN,n_neighbors)
            XNN_0_L[j,:,:]=XNN_0.float();XNN_1_L[j,:,:]=XNN_1.float()
        XNN_0_T =XNN_0_L[:Tr,:,:].float(); XNN_1_T =XNN_1_L[:Tr,:,:].float()
        XNN_0_V =XNN_0_L[Tr:,:,:].float(); XNN_1_V =XNN_0_L[Tr:,:,:].float()     
        X_dict0_T[f"X{i}"]=XNN_0_T;X_dict1_T[f"X{i}"]=XNN_1_T#N*L*NN*E
        X_dict0_V[f"X{i}"]=XNN_0_V;X_dict1_V[f"X{i}"]=XNN_1_V
        n_dataset0_T.tensors += (X_dict0_T[f"X{i}"],)
        n_dataset1_T.tensors += (X_dict1_T[f"X{i}"],)
        n_dataset0_V.tensors += (X_dict0_V[f"X{i}"],)
        n_dataset1_V.tensors += (X_dict1_V[f"X{i}"],)
    return X_0_T, X_1_T, X_0_V, X_1_V, X_dict0_T, X_dict1_T, n_dataset0_T, n_dataset1_T,X_dict0_V, X_dict1_V, n_dataset0_V, n_dataset1_V

def generate_embedd_data0(data,embedding_dim, time_delay,n_neighbors,rate):
    ##here _0 represents "t", _1 represents "t-1".
    ## _T is the samples for training, _V is the samples for validation.
    L=data.shape[0]-embedding_dim+1-1
    Tr=round(L*rate)
    NN=n_neighbors
    E=embedding_dim
    X_0_T = {};X_1_T = {};X_0_V = {};X_1_V = {}
    X_0 = {};X_1 = {} 
    for i in range(data.shape[1]):
        x_delay=delay_embedding(data[:,i], embedding_dim, time_delay)
        x_delay0 = x_delay[:-1, :]; x_delay1 = x_delay[1:, :]#t-1和t
        #x_delay0 = x_delay; x_delay1 = x_delay
        normalizer_x0 = MinMaxNormalize(x_delay0); normalizer_x1 = MinMaxNormalize(x_delay1)
        n_x0 = normalizer_x0.normalize(x_delay0); n_x1 = normalizer_x1.normalize(x_delay1)
        n_x0 =torch.from_numpy(n_x0); n_x1 =torch.from_numpy(n_x1)
        #n_x00 =n_x0.unsqueeze(1).repeat(1, n_neighbors, 1);n_x11 =n_x1.unsqueeze(1).repeat(1, n_neighbors, 1)
        #X_0_T[f"X{i}"] =n_x00[:Tr,:,:].float(); X_1_T[f"X{i}"] =n_x11[:Tr,:,:].float()
        #X_0_V[f"X{i}"] =n_x00[Tr:,:,:].float(); X_1_V[f"X{i}"] =n_x11[Tr:,:,:].float()
        X_0_T[f"X{i}"] =n_x0[:Tr,:].float(); X_1_T[f"X{i}"] =n_x1[:Tr,:].float()
        X_0_V[f"X{i}"] =n_x0[Tr:,:].float(); X_1_V[f"X{i}"] =n_x1[Tr:,:].float()
        X_0[f"X{i}"] =n_x0.float(); X_1[f"X{i}"] =n_x1.float()
        #X_0[i] = n_x0.float(); X_1[i] = n_x1.float()
        
    return X_0_T, X_1_T, X_0_V, X_1_V, X_0,X_1

def generate_embedd_data(data,embedding_dim, time_delay,n_neighbors):
    ##here _0 represents "t", _1 represents "t-1".
    ## _T is the samples for training, _V is the samples for validation.
    L=data.shape[0]-embedding_dim+1-1
    Tr=round(L*0.7)
    NN=n_neighbors
    E=embedding_dim
    X_0_T = {};X_1_T = {};X_0_V = {};X_1_V = {}
    X_0 = {};X_1 = {} 
    for i in range(data.shape[1]):
        x_delay=delay_embedding(data[:,i], embedding_dim, time_delay)
        x_delay0 = x_delay[:-1, :]; x_delay1 = x_delay[1:, :]#t-1和t
        #x_delay0 = x_delay; x_delay1 = x_delay
        normalizer_x0 = MinMaxNormalize(x_delay0); normalizer_x1 = MinMaxNormalize(x_delay1)
        n_x0 = normalizer_x0.normalize(x_delay0); n_x1 = normalizer_x1.normalize(x_delay1)
        n_x0 =torch.from_numpy(n_x0); n_x1 =torch.from_numpy(n_x1)
        #n_x00 =n_x0.unsqueeze(1).repeat(1, n_neighbors, 1);n_x11 =n_x1.unsqueeze(1).repeat(1, n_neighbors, 1)
        #X_0_T[f"X{i}"] =n_x00[:Tr,:,:].float(); X_1_T[f"X{i}"] =n_x11[:Tr,:,:].float()
        #X_0_V[f"X{i}"] =n_x00[Tr:,:,:].float(); X_1_V[f"X{i}"] =n_x11[Tr:,:,:].float()
        X_0_T[f"X{i}"] =n_x0[:Tr,:].float(); X_1_T[f"X{i}"] =n_x1[:Tr,:].float()
        X_0_V[f"X{i}"] =n_x0[Tr:,:].float(); X_1_V[f"X{i}"] =n_x1[Tr:,:].float()
        X_0[f"X{i}"] =n_x0.float(); X_1[f"X{i}"] =n_x1.float()
        #X_0[i] = n_x0.float(); X_1[i] = n_x1.float()
        
    return X_0_T, X_1_T, X_0_V, X_1_V, X_0,X_1

def generate_KNN_data(data,X0,Y1,embedding_dim, n_neighbors):
    L=data.shape[0]-embedding_dim+1-1
    Tr=round(L*0.7)
    Vr=L-Tr
    NN=n_neighbors
    E=embedding_dim
    XNN_L = torch.zeros(L, NN, E); YNN_L = torch.zeros(L, NN, E)
    XNN_T = torch.zeros(Tr, NN, E); YNN_T = torch.zeros(Tr, NN, E)
    XNN_V = torch.zeros(Vr, NN, E); YNN_V = torch.zeros(Vr, NN, E)
    n_datasetX_T = TensorDataset();n_datasetY_T = TensorDataset()
    n_datasetX_V = TensorDataset();n_datasetY_V = TensorDataset()    
    for j in range(L): 
        Y_KNN = torch.cat([Y1[:j], Y1[j+1:]], dim=0);#
        X_KNN = torch.cat([X0[:j], X0[j+1:]], dim=0)   
        XNN, YNN=KNN1(Y1[j],Y_KNN, X_KNN, n_neighbors)
        XNN=XNN-X0[j]; YNN=YNN-Y1[j]
        #XNN_0=KNN(n_x0[j],X_0_KNN,n_neighbors)
        XNN_L[j,:,:]=XNN.float();YNN_L[j,:,:]=YNN.float()
    XNN_T =XNN_L[:Tr,:,:].float(); YNN_T =YNN_L[:Tr,:,:].float()
    XNN_V =XNN_L[Tr:,:,:].float(); YNN_V =YNN_L[Tr:,:,:].float()     
    n_datasetX_T.tensors = (XNN_T,)
    n_datasetY_T.tensors = (YNN_T,)
    n_datasetX_V.tensors = (XNN_V,)
    n_datasetY_V.tensors = (YNN_V,)
    return  XNN_T, YNN_T, XNN_V, YNN_V

def delay_embedding(data, embedding_dim, lag):
    data = np.array(data)
    n = data.shape[0]
    m = n - (embedding_dim - 1) * lag

    embedded_data = np.zeros((m, embedding_dim))
  
    for i in range(m):
        for j in range(embedding_dim):
            embedded_data[i, j] = data[i+embedding_dim - 1 - (j) * lag]
    #embedded_data = embedded_data[:, ::-1]
    return embedded_data

class MinMaxNormalize:
    """
    Minmax normalization for a torch tensor to (-1, 1)

    Parameters
    ----------
    tensor: torch.Tensor
        The tensor to be normalized
    dim: int
        The dimension to be normalized
    """

    def __init__(self, x, dim=0, dtype="float64"):
        self.dim = dim
        self.dtype = dtype
        x = x.astype(self.dtype)
        self.min = np.min(x, axis=dim, keepdims=True)
        self.max = np.max(x, axis=dim, keepdims=True)

    def normalize(self, x):
        x = x.astype(self.dtype)
        for i in range(x.shape[1]):
            if self.max[:,i] == self.min[:,i]:
                x[:,i]=x[:,i]  # 返回一个全为 0 的数组
            else:
                x[:,i]=2 * (x[:,i] - self.min[:,i]) / (self.max[:,i] - self.min[:,i]) - 1
        return x

    def denormalize(self, x):
        x = x.astype(self.dtype)
        return (x + 1) / 2 * (self.max - self.min) + self.min

def KNN1(y, Y, X, n_neighbors):
    #look for the neighbor of yt, tk, in Y
    #return the the map tk in X
    # 将 y 和 Y 转换为 NumPy 数组
    y_np = y.cpu().numpy()  # 确保在 CPU 上
    Y_np = Y.cpu().numpy()  # 确保在 CPU 上
    X_np = X.cpu().numpy()  # 确保在 CPU 上
    # 查找邻居
    knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto')
    knn.fit(Y_np)
    distances, indices = knn.kneighbors([y_np])
    YNN = Y[indices[0]]
    XNN = X[indices[0]]
    return XNN, YNN

def KNN(x,X,n_neighbors):
    knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto')
    knn.fit(X.numpy())
    distances, indices = knn.kneighbors([x.numpy()])
    XNN = X[indices[0]]
    return XNN

def Normalize(x,y):
    x0=x*x/(x*x+y*y)
    y0=y*y/(x*x+y*y)
    if torch.isnan(x0) or torch.isnan(y0):
        x0=x/(x+y)
        y0=y/(x+y)
    return x0, y0

def cal_kld_loss(mean, logvar):
    return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1).mean()

def Ortho_loss(x, y):
    cosine_similarity = torch.nn.functional.cosine_similarity(x, y, dim=1)
    orthogonality_loss = torch.mean((cosine_similarity)**2)
    return orthogonality_loss

def Confounder(Net):
    N=Net.shape[1]
    Updata_Net=Net.clone()
    Net_confd=torch.zeros(N,N)
    for i in range(N):
        for j in range(N):
            if i!=j:
                for k in range(N):
                    if k!=j:
                        if Net[i,j]==1:
                            if Net[i,k] == 1: 
                                Net_confd[j,k]=i
                                if Net[j,k]!=1:
                                    Updata_Net[j,k]=2   
                                if Net[k,j]==0:
                                    Updata_Net[k,j]=2                               
    return Updata_Net,Net_confd 

def CausalNet_judge_CIC(causal_index,Net_causal): 
    N=Net_causal.shape[0]
    M=Net_causal.shape[1]
    for i in range(N):
        for j in range(M):
            if i != j:         
                if causal_index[i,j]<=0.25:
                    Net_causal[i,j]=0
                elif 0.25 < causal_index[i, j] < 0.75:
                    if causal_index[j,i]>=0.75:
                        Net_causal[i,j]=0 
                    else:
                        Net_causal[i,j]=2   
                elif causal_index[i,j]>=0.75:
                    Net_causal[i,j]=1
    return Net_causal

def CausalNet_judge_CIC_Single(causal_index,Net_causal): 
    N=Net_causal.shape[0]
    M=Net_causal.shape[1]
    for i in range(N):
        for j in range(M):
            if i != j:         
                if causal_index[i,j]<=0.25:
                    Net_causal[i,j]=0
                elif 0.25 < causal_index[i, j] < 0.75:
                        Net_causal[i,j]=2   
                elif causal_index[i,j]>=0.75:
                    Net_causal[i,j]=1
    return Net_causal

def CausalNet_judge_IntCIC(causal_index):        
    if causal_index>=0.5:
        Net_causal=1
    else:
        Net_causal=0
    return Net_causal

def objective_function(b,x,y):
    return np.linalg.norm(y - (b[0] * (x ** b[1])))

def normalize_0_1(data):
    # 计算最小值和最大值
    min_value = np.min(data)
    max_value = np.max(data)
    # 处理特殊情况：所有值相同
    if max_value == min_value:
        return np.zeros_like(data)  # 返回一个全为 0 的数组
    # 进行 0-1 标准化
    normalized_data = (data - min_value) / (max_value - min_value)
    return normalized_data

def subnet_adjac(edges, sub_network_nodes):
    # 创建一个从大网络到子网络的映射
    node_index_map = {node: idx for idx, node in enumerate(sub_network_nodes)}
    # 创建邻接矩阵
    sub_network_size = len(sub_network_nodes)
    sub_adjacency_matrix = np.zeros((sub_network_size, sub_network_size), dtype=int)
    # 填充子网络的邻接矩阵
    for start, end in edges:
        if start in node_index_map and end in node_index_map:
            sub_start = node_index_map[start]
            sub_end = node_index_map[end]
            sub_adjacency_matrix[sub_start, sub_end] = 1
    return sub_adjacency_matrix

def normalize_kld(kld):
    # 初始化一个与 KLD 相同形状的数组来存储标准化后的结果
    normalized_kld = np.zeros_like(kld)
    # 对第三维的每个切片进行标准化
    for i in range(kld.shape[2]):  # 遍历第三维的每个切片
        sKLD = kld[:, :, i]  # 取出第 i 个切片       
        # 计算最小值和最大值
        min_val = sKLD.min()
        max_val = sKLD.max()       
        # 进行 0-1 标准化
        if max_val - min_val > 0:
            normalized_kld[:, :, i] = (sKLD - min_val) / (max_val - min_val)
        else:
            normalized_kld[:, :, i] = 0  # 如果所有值相同，设置为 0
    return normalized_kld

def move_to_device(data, device):
    if isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}
    elif isinstance(data, list):
        return [move_to_device(item, device) for item in data]
    elif isinstance(data, tuple):
        return tuple(move_to_device(item, device) for item in data)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data  # 对于不需要处理的类型，直接返回

