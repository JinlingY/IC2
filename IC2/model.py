import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from IC2.utils import cal_kld_loss,Ortho_loss#,Normalize
import matplotlib.pyplot as plt

def Normalize(x,y):
    x0=x*x/(x*x+y*y)
    y0=y*y/(x*x+y*y)
    if torch.isnan(x0) or torch.isnan(y0):
        x0=x/(x+y)
        y0=y/(x+y)
    return x0, y0
   
class MLP_CIC(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims,
                 act_fn=nn.ReLU(), batch_norm=False):
        super(MLP_CIC, self).__init__()

        # Initialize a list to store the layers
        layers = []

        # Calculate the dimensions for the layers
        dims = [in_dim] + hidden_dims + [out_dim]

        # Create the layers based on the dimensions
        for i in range(1, len(dims)):
            layers.append(nn.Linear(dims[i - 1], dims[i]))  # Linear layer
            if batch_norm and i < len(dims) - 1:  # Batch normalization (except for the output layer)
                layers.append(nn.BatchNorm1d(dims[i]))
            if i < len(dims) - 1:  # Activation function (except for the output layer)
                layers.append(act_fn)

        # Define the neural network as a sequence of layers
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # Forward pass through the network
        return self.mlp(x)
   
class MLP_IntCIC(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims,n_neighbors,
                 act_fn=nn.ReLU(), batch_norm=False):
        super(MLP_IntCIC, self).__init__()

        # Initialize a list to store the layers
        layers = []

        # Calculate the dimensions for the layers
        dims = [in_dim] + hidden_dims + [out_dim]

        # Create the layers based on the dimensions
        for i in range(1, len(dims)):
            layers.append(nn.Linear(dims[i - 1], dims[i]))  # Linear layer
            if batch_norm and i < len(dims) - 1:  # Batch normalization (except for the output layer)
                layers.append(nn.BatchNorm1d(n_neighbors))
            if i < len(dims) - 1:  # Activation function (except for the output layer)
                layers.append(act_fn)

        # Define the neural network as a sequence of layers
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # Forward pass through the network
        return self.mlp(x)

class VAEOutput:
    def __init__(self, mean, logvar, sample, recon=None):
        self.mean = mean
        self.logvar = logvar
        self.sample = sample
        self.recon = recon

    def set_latent_zero(self, zero_vars):
        self.mean = set_zero_vars(self.mean, zero_vars)
        self.logvar = set_zero_vars(self.logvar, zero_vars)
        self.sample = set_zero_vars(self.sample, zero_vars)

    def copy(self):
        # deep copy the object consider the recon is None
        if self.recon is None:
            return VAEOutput(self.mean.copy(), self.logvar.copy(), self.sample.copy())
        else:
            return VAEOutput(self.mean.copy(), self.logvar.copy(), self.sample.copy(), self.recon.copy())

    def to_numpy(self):
        # convert the output to numpy
        def _to_numpy(x):
            if isinstance(x, np.ndarray):
                return x
            else:
                return x.detach().cpu().numpy()

        return VAEOutput({key: _to_numpy(self.mean[key]) for key in self.mean.keys()},
                         {key: _to_numpy(self.logvar[key]) for key in self.logvar.keys()},
                         {key: _to_numpy(self.sample[key]) for key in self.sample.keys()},
                         {key: _to_numpy(self.recon[key]) for key in self.recon.keys()})

def set_zero_vars(input_dict, zero_vars=None):
    """
    Set values in the input_dict to zero based on the keys in zero_vars.

    Args:
        input_dict (dict): A dictionary where keys represent variable names and values are torch tensors.
        zero_vars (list): A list of variable names to set to zero. If None, no variables are set to zero.

    Returns:
        dict: A modified dictionary with selected variables set to zero on the same device as the input tensors.
    """
    if zero_vars is None:
        return input_dict

    assert all(var_name in input_dict for var_name in zero_vars), "All zero_vars must exist in the input_dict."

    result_dict = {}
    for var_name, var_value in input_dict.items():
        if var_name in zero_vars:
            zero_tensor = torch.zeros_like(var_value, device=var_value.device)
            result_dict[var_name] = zero_tensor
        else:
            result_dict[var_name] = var_value

    return result_dict

##CIC

class VAE_Encoder_CIC(nn.Module):
    def __init__(self, in_dim: int,
                 out_dim: int,
                 hidden_dims=[50, 50],
                 latent_dims= 32,
                 batch_norm: bool = True,
                 act_fn: nn.Module = nn.ReLU(),
                  ):
        super(VAE_Encoder_CIC, self).__init__()
        self.encoder = nn.Sequential(
            MLP_CIC(in_dim, out_dim, hidden_dims, batch_norm=batch_norm, act_fn=act_fn)
        )
        
        self.fc_mu = nn.Linear(out_dim, out_dim)
        self.fc_logvar = nn.Linear(out_dim, out_dim)       
        
    def forward(self, x):
        #print(x.shape)
        z = self.encoder(x)  
        #print(z.shape)
        mu = self.fc_mu(z)
        logvar = self.fc_logvar(z)
        return mu, logvar#, z

class VAE_Decoder_CIC(nn.Module):
    def __init__(self, in_dim: int,
                 out_dim: int,
                 hidden_dims=[50, 50],
                 latent_dims= 32,
                 batch_norm: bool = True,
                 act_fn: nn.Module = nn.ReLU(),
                  ):
        super(VAE_Decoder_CIC, self).__init__()
        self.decoder = nn.Sequential(
            MLP_CIC(in_dim, out_dim, hidden_dims, batch_norm=batch_norm, act_fn=act_fn)        
        )       
    def forward(self, z):     
        #print(z.shape)
        recon_x = self.decoder(z)  
        #print(recon_x.shape)
        return recon_x

class DualVAE(nn.Module):
    """
    Causal Autoencoder that decompose the delay embeding of varaible x and y into shared and private latent variables.
    """
    
    def __init__(self, x_dim: int,
                 y_dim: int,
                 zx_dim: int = 2,
                 zy_dim: int = 2,
                 s_dim: int = 2,
                 hidden_dims=[50, 50],
                 act_fn: nn.Module = nn.ReLU(),
                 batch_norm: bool = True,
                 ):
        super(DualVAE, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.zx_dim = zx_dim
        self.zy_dim = zy_dim
        self.s_dim = s_dim
        self.vae_X_encoder = VAE_Encoder_CIC(x_dim, zx_dim + s_dim, hidden_dims, batch_norm=batch_norm, act_fn=act_fn)
        #self.vae_X_encoder_logvar = VAE_Encoder4_1(x_dim, zx_dim + s_dim, hidden_dims, batch_norm=batch_norm, act_fn=act_fn)
        self.vae_X_decoder = VAE_Decoder_CIC(zx_dim + s_dim, x_dim, hidden_dims[::-1], batch_norm=batch_norm, act_fn=act_fn)
        #self.vae_X_decoder = VAE_Decoder(zx_dim + s_dim, x_dim, hidden_dims[::-1], batch_norm=batch_norm, act_fn=act_fn)
        self.vae_Y_encoder = VAE_Encoder_CIC(y_dim, zy_dim  + s_dim, hidden_dims, batch_norm=batch_norm, act_fn=act_fn)
        #self.vae_Y_encoder_logvar = VAE_Encoder4_1(y_dim, zy_dim  + s_dim, hidden_dims, batch_norm=batch_norm, act_fn=act_fn)
        self.vae_Y_decoder = VAE_Decoder_CIC(zy_dim + s_dim, y_dim, hidden_dims[::-1], batch_norm=batch_norm, act_fn=act_fn)
        #self.vae_Y_decoder = VAE_Decoder(zy_dim + s_dim, y_dim, hidden_dims[::-1], batch_norm=batch_norm, act_fn=act_fn)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)+1e-7
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x, y, zero_vars=None):
        mu_X, logvar_X= self.vae_X_encoder(x)  
        #logvar_X = self.vae_X_encoder_logvar(x)     
        z_x_mean, s_x_mean = torch.split(mu_X, [self.zx_dim, self.s_dim], dim=1)
        z_x_logvar, s_x_logvar = torch.split(logvar_X, [self.zx_dim, self.s_dim], dim=1)
        
        mu_Y, logvar_Y= self.vae_Y_encoder(y)  
        #logvar_Y = self.vae_Y_encoder_logvar(y) 
        z_y_mean, s_y_mean = torch.split(mu_Y, [self.zy_dim, self.s_dim], dim=1)
        z_y_logvar, s_y_logvar = torch.split(logvar_Y, [self.zy_dim, self.s_dim], dim=1)
        
        mean = {"z_x": z_x_mean, "z_y": z_y_mean, "s_x": s_x_mean, "s_y": s_y_mean}
        logvar = {"z_x": z_x_logvar, "z_y": z_y_logvar, "s_x": s_x_logvar, "s_y": s_y_logvar}
        sample = {key: self.reparameterize(mean[key], logvar[key]) for key in mean.keys()}
        output = VAEOutput(mean, logvar, sample)
        
        zx_zeros = torch.zeros_like(output.sample["z_x"])
        zy_zeros = torch.zeros_like(output.sample["z_y"])
        s_zeros = torch.zeros_like(output.sample["s_x"])
        output.set_latent_zero(zero_vars)
        recon_X = self.vae_X_decoder(torch.cat([output.sample["z_x"], output.sample["s_x"]], dim=1))# 通过 z1_X 和 z2_X 重构得到 recon_X          
        recon_X_s = self.vae_X_decoder(torch.cat([zx_zeros, output.sample["s_x"]], dim=1))# 通过 z2_X 重构得到 recon_X_z2
        recon_X_zx = self.vae_X_decoder(torch.cat([output.sample["z_x"], s_zeros], dim=1))# 通过 z2_X 重构得到 recon_X_z2
        recon_X_s1 = self.vae_X_decoder(torch.cat([zx_zeros, output.sample["s_y"]], dim=1))# 通过 z2_X 重构得到 recon_X_z2
        recon_Y = self.vae_Y_decoder(torch.cat([output.sample["z_y"], output.sample["s_y"]], dim=1))# 通过 z1_X 和 z2_X 重构得到 recon_X          
        recon_Y_s = self.vae_Y_decoder(torch.cat([zy_zeros, output.sample["s_y"]], dim=1))# 通过 z2_X 重构得到 recon_X_z2
        recon_Y_zy = self.vae_Y_decoder(torch.cat([output.sample["z_y"], s_zeros], dim=1))
        output.recon = {"x": recon_X, "y": recon_Y, "x_s": recon_X_s,  "x_s1": recon_X_s1,"x_zx": recon_X_zx, "y_s": recon_Y_s,  "y_zy": recon_Y_zy} #, "x_s1": recon_X_s1}  
        return output
   
def train_dualvae(model, weights, x_delay, y_delay, device, num_epochs, batch_size, learning_rate):
    dataset = TensorDataset(x_delay.float(), y_delay.float())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)#,drop_last=True)
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    #pbar = tqdm(range(num_epochs))
    losses = [];recon_X_losses= [];recon_Y_losses= []; recon_X_s_losses= []; ortho_losses= []; equiv_losses= []
    for epoch in range(num_epochs):
        for x, y in dataloader:
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)
            output = model.forward(x, y)            
            loss_dict = dualvae_loss(x, y, output,weights)
            loss = loss_dict["total_loss"]            
            loss.backward()
            optimizer.step()
        lr_scheduler.step(loss)

        losses.append(loss)
        recon_X_losses.append(loss_dict["recon_X_loss"])
        recon_Y_losses.append(loss_dict["recon_Y_loss"])
        recon_X_s_losses.append(loss_dict["recon_X_s_loss"])
        ortho_losses.append(loss_dict["ortho_loss"])
        equiv_losses.append(loss_dict["equiv_loss"])
    steps = range(1, len(losses) + 1)

    # only for data tranformation
    losses_np = [loss.detach().cpu().numpy() for loss in losses]
    recon_X_losses_np = [loss_dict["recon_X_loss"].detach().cpu().numpy() for loss_dict["recon_X_loss"] in recon_X_losses]
    recon_Y_losses_np = [loss_dict["recon_Y_loss"].detach().cpu().numpy() for loss_dict["recon_Y_loss"] in recon_Y_losses]
    recon_X_s_losses_np = [loss_dict["recon_X_s_loss"].detach().cpu().numpy() for loss_dict["recon_X_s_loss"] in recon_X_s_losses]
    ortho_losses_np = [loss_dict["ortho_loss"].detach().cpu().numpy() for loss_dict["ortho_loss"] in ortho_losses]
    equiv_losses_np = [loss_dict["equiv_loss"].detach().cpu().numpy() for loss_dict["equiv_loss"] in equiv_losses]
    return model

def dualvae_loss(x, y, output,weights):
    #1 重构损失
    recon_X_loss = nn.functional.mse_loss(output.recon["x"], x)#JS_function1
    recon_Y_loss = nn.functional.mse_loss(output.recon["y"], y)
    recon_X_s_loss = nn.functional.mse_loss(output.recon["x_s"], x)
    #3
    ortho_loss_1 = Ortho_loss(output.sample["z_x"], output.sample["z_y"])
    ortho_loss_2 = Ortho_loss(output.sample["z_x"], output.sample["s_x"])
    ortho_loss_3 = Ortho_loss(output.sample["z_y"], output.sample["s_y"])
    ortho_loss = ortho_loss_1 + ortho_loss_2 + ortho_loss_3
    #4
    equiv_loss1 = nn.functional.mse_loss(output.sample["s_x"], output.sample["s_y"])
    equiv_loss2 = nn.functional.mse_loss(output.recon["x_s1"], output.recon["x_s"])
    equiv_loss = equiv_loss1+equiv_loss2

    #5
    l1_loss=0
    kld_loss=0
    for key in output.mean.keys():
        kld_loss += cal_kld_loss(output.mean[key], output.logvar[key])
        l1_loss += torch.mean(torch.abs(output.sample[key]))#
    total_loss = weights[0] * recon_X_loss + weights[1] * recon_Y_loss + weights[2] * recon_X_s_loss+ weights[3] * ortho_loss+ weights[4] * equiv_loss+ weights[5] * kld_loss + weights[6] * l1_loss  
            
    #total_loss = 0.35*recon_X_loss + 0.35*recon_Y_loss + 0.13*recon_X_s_loss + 0.001*kld_loss + 0.001*l1_loss  + 0.09*ortho_loss + 0.14*equiv_loss 
    loss_dict = {"total_loss": total_loss, "recon_X_loss": recon_X_loss, "recon_Y_loss": recon_Y_loss,  "recon_X_s_loss": recon_X_s_loss, #"recon_X_s1_loss": recon_X_s1_loss, 
                 #"recon_X_zx_loss": recon_X_zx_loss, #"recon_Y_s_loss": recon_Y_s_loss, "recon_Y_zy_loss": recon_Y_zy_loss, 
                 #"mi_diff_X": mi_diff_X,
                 #"NMI_xy": NMI_xy,"MI_xy": MI_xy, "H_x": H_x, "H_y": H_y, 
                 "kld_loss": kld_loss, "l1_loss": l1_loss, "ortho_loss": ortho_loss, "equiv_loss": equiv_loss}
    return loss_dict

def evaluate_CIC(model, weights,x,y, device, return_numpy=True):
    """
    reconstruct the data and plot the reconstruction vs. the original data
    """
    
    #x = dataset.tensors[0].to(device)
    #y = dataset.tensors[1].to(device)
    x =x.to(device)
    y =y.to(device)
    model.eval().to(device)
    with torch.no_grad():
         output = model(x, y)
         loss_dict = dualvae_loss(x, y, output, weights)
         loss = loss_dict["total_loss"]
    #print("loss: {:.4f}, recon_X_loss: {:.4f}, recon_Y_loss: {:.4f}, recon_X_s_loss: {:.4f}, ortho_loss: {:.4f}, equiv_loss: {:.4f}, kld_loss: {:.4f}, l1_loss: {:.4f}".format(  #recon_Y_s_loss: {:.4f}, recon_X_s1_loss: {:.4f},  mi_diff_Y: {:.4f}, NMI_xy: {:.4f}, MI_xy: {:.4f}, H_x: {:.4f}, H_y: {:.4f}, 
             #loss.item(), loss_dict["recon_X_loss"].item(), loss_dict["recon_Y_loss"].item(), loss_dict["recon_X_s_loss"].item(),  #recon_X_zx_loss: {:.4f}, mi_diff_X: {:.4f}, equiv_loss: {:.4f}, 
             #loss_dict["ortho_loss"].item(), loss_dict["equiv_loss"].item(), 
             #loss_dict["kld_loss"].item(), loss_dict["l1_loss"].item()))
         
    recon_x = output.recon["x"]
    recon_y = output.recon["y"]
    recon_x_s = output.recon["x_s"]
    recon_y_s = output.recon["y_s"]

    recon_x_zx = output.recon["x_zx"]
    recon_y_zy = output.recon["y_zy"]
    recon_x_s1 = output.recon["x_s1"].detach().cpu().numpy()

    ##MSE
    MSE_X61 = nn.functional.mse_loss(x,output.recon["x_s"])
    MSE_X71 = nn.functional.mse_loss(x,output.recon["x_zx"])
    MSE_X62,MSE_X72 = Normalize(MSE_X61,MSE_X71)

    ##MSE
    #print("重构X_s与原始X的MSE:", MSE_X62)
    #print("重构X_zx与原始X的MSE:", MSE_X72)

    #only for GPU
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    recon_x = recon_x.detach().cpu().numpy()
    recon_x_s = recon_x_s.detach().cpu().numpy()
    recon_x_zx = recon_x_zx.detach().cpu().numpy()
    recon_y = recon_y.detach().cpu().numpy()
    recon_y_s = recon_y_s.detach().cpu().numpy()
    recon_y_zy = recon_y_zy.detach().cpu().numpy()
    return output,MSE_X62,MSE_X72    

##IntCIC
class VAE_Encoder_IntCIC(nn.Module):
    def __init__(self, in_dim: int,
                 out_dim: int,
                 hidden_dims=[50, 50, 50],
                 n_neighbors=10,
                 latent_dims= 32,
                 batch_norm: bool = True,
                 act_fn: nn.Module = nn.ReLU(),
                  ):
        super(VAE_Encoder_IntCIC, self).__init__()
        self.encoder = nn.Sequential(
            MLP_IntCIC(in_dim, out_dim, hidden_dims,n_neighbors, batch_norm=batch_norm, act_fn=act_fn)
        )
        
        self.fc_mu = nn.Linear(out_dim, out_dim)
        self.fc_logvar = nn.Linear(out_dim, out_dim)       
        
    def forward(self, x):
        #print(x.shape)
        z = self.encoder(x)  
        #print(z.shape)
        mu = self.fc_mu(z)
        logvar = self.fc_logvar(z)
        return mu, logvar#, z

class VAE_Decoder_IntCIC(nn.Module):
    def __init__(self, in_dim: int,
                 out_dim: int,
                 hidden_dims=[50, 50],
                 n_neighbors=10,
                 latent_dims= 32,
                 batch_norm: bool = True,
                 act_fn: nn.Module = nn.ReLU(),
                  ):
        super(VAE_Decoder_IntCIC, self).__init__()
        self.decoder = nn.Sequential(
            MLP_IntCIC(in_dim, out_dim, hidden_dims,n_neighbors, batch_norm=batch_norm, act_fn=act_fn)        
        )       
    def forward(self, z):     
        #print(z.shape)
        recon_x = self.decoder(z)  
        #print(recon_x.shape)
        return recon_x

class TripleVAE(nn.Module):
    """
    Causal Autoencoder that decompose the delay embeding of varaible x and y into shared and private latent variables.
    """
    
    def __init__(self, dx_dim: int,
                 dy_dim: int,
                 dzx_dim: int = 2,
                 dzy_dim: int = 2,
                 ds_dim: int = 2,
                 hidden_dims=[50, 50],
                 n_neighbors: int = 2,
                 act_fn: nn.Module = nn.ReLU(),
                 batch_norm: bool = True,
                 ):
        super(TripleVAE, self).__init__()
        self.dx_dim = dx_dim
        self.dy_dim = dy_dim
        self.dzx_dim = dzx_dim
        self.dzy_dim = dzy_dim
        self.ds_dim = ds_dim#
        #1
        #self.vae_dX_encoder = VAE_Encoder(dx_dim, dzx_dim + ds_dim, hidden_dims,n_neighbors, batch_norm=batch_norm, act_fn=act_fn)
        #self.vae_dX_decoder = VAE_Decoder(dzx_dim + ds_dim , dx_dim, hidden_dims[::-1],n_neighbors, batch_norm=batch_norm, act_fn=act_fn)
        #self.vae_dY_encoder = VAE_Encoder(dy_dim, dzy_dim  + ds_dim, hidden_dims,n_neighbors, batch_norm=batch_norm, act_fn=act_fn)
        #self.vae_dY_decoder = VAE_Decoder(dzy_dim + ds_dim, dy_dim, hidden_dims[::-1],n_neighbors, batch_norm=batch_norm, act_fn=act_fn)
        #2
        self.vae_dX_encoder = VAE_Encoder_IntCIC(dx_dim, dzx_dim+ ds_dim , hidden_dims,n_neighbors, batch_norm=batch_norm, act_fn=act_fn)
        self.vae_dX_decoder = VAE_Decoder_IntCIC(dzx_dim + ds_dim, dx_dim, hidden_dims[::-1],n_neighbors, batch_norm=batch_norm, act_fn=act_fn)
        self.vae_dY_encoder = VAE_Encoder_IntCIC(dy_dim, dzy_dim+ ds_dim , hidden_dims,n_neighbors, batch_norm=batch_norm, act_fn=act_fn)
        self.vae_dY_decoder = VAE_Decoder_IntCIC(dzy_dim + ds_dim, dy_dim, hidden_dims[::-1],n_neighbors, batch_norm=batch_norm, act_fn=act_fn)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)+1e-7
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, dx, dy, zero_vars=None):
        ##1
        mu_dX, logvar_dX= self.vae_dX_encoder(dx)  
        dz_x_mean, ds_x_mean = torch.split(mu_dX, [self.dzx_dim, self.ds_dim], dim=2)
        dz_x_logvar, ds_x_logvar = torch.split(logvar_dX, [self.dzx_dim, self.ds_dim], dim=2)
        
        mu_dY, logvar_dY= self.vae_dY_encoder(dy)  
        dz_y_mean, ds_y_mean = torch.split(mu_dY, [self.dzy_dim, self.ds_dim], dim=2)
        dz_y_logvar, ds_y_logvar = torch.split(logvar_dY, [self.dzy_dim, self.ds_dim], dim=2)
        
        mean = {"dz_x": dz_x_mean, "dz_y": dz_y_mean, "ds_x": ds_x_mean, "ds_y": ds_y_mean}
        logvar = {"dz_x": dz_x_logvar, "dz_y": dz_y_logvar, "ds_x": ds_x_logvar, "ds_y": ds_y_logvar}
        sample = {key: self.reparameterize(mean[key], logvar[key]) for key in mean.keys()}
        output = VAEOutput(mean, logvar, sample)
        ##2       
        #mean = {"dz_x": mu_dX, "dz_y": mu_dY, "ds_x": mu_dX, "ds_y": mu_dY}
        #logvar = {"dz_x": logvar_dX, "dz_y": logvar_dY, "ds_x": logvar_dX, "ds_y": logvar_dY}
        #sample = {key: self.reparameterize(mean[key], logvar[key]) for key in mean.keys()}
        #output = VAEOutput(mean, logvar, sample)

        
        dzx_zeros = torch.zeros_like(output.sample["dz_x"])
        dzy_zeros = torch.zeros_like(output.sample["dz_y"])
        ds_zeros = torch.zeros_like(output.sample["ds_x"])
        output.set_latent_zero(zero_vars)
        recon_dX = self.vae_dX_decoder(torch.cat([output.sample["dz_x"], output.sample["ds_x"]], dim=2))# 通过 z1_X 和 z2_X 重构得到 recon_X          
        recon_dX_s = self.vae_dX_decoder(torch.cat([dzx_zeros, output.sample["ds_x"]], dim=2))# 通过 z2_X 重构得到 recon_X_z2
        recon_dX_zx = self.vae_dX_decoder(torch.cat([output.sample["dz_x"], ds_zeros], dim=2))# 通过 z2_X 重构得到 recon_X_z2
        recon_dX_s1 = self.vae_dX_decoder(torch.cat([dzx_zeros, output.sample["ds_y"]], dim=2))# 通过 z2_X 重构得到 recon_X_z2
        ##
        recon_dY = self.vae_dY_decoder(torch.cat([output.sample["dz_y"], output.sample["ds_y"]], dim=2))# 通过 z1_X 和 z2_X 重构得到 recon_X          
        recon_dY_s = self.vae_dY_decoder(torch.cat([dzy_zeros, output.sample["ds_y"]], dim=2))# 通过 z2_X 重构得到 recon_X_z2
        recon_dY_zy = self.vae_dY_decoder(torch.cat([output.sample["dz_y"], ds_zeros], dim=2))
        output.recon = {"dx": recon_dX, "dy": recon_dY, "dx_s": recon_dX_s,  "dx_s1": recon_dX_s1,"dx_zx": recon_dX_zx, "dy_s": recon_dY_s,  "dy_zy": recon_dY_zy} #, "x_s1": recon_X_s1}  
        return output

def train_triplevae(model, weights, dx, dy, device, num_epochs, batch_size, learning_rate):
    dataset = TensorDataset(dx.float(), dy.float())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)#,drop_last=True)
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    #pbar = tqdm(range(num_epochs))
    losses = [];recon_dX_losses= [];recon_dY_losses= [];  recon_dX_s_losses= []; ortho_losses= []; equiv_losses= []
    for epoch in range(num_epochs):
        for dx, dy in dataloader:
            optimizer.zero_grad()
            dx = dx.to(device)
            dy = dy.to(device)
            output = model.forward(dx, dy)            
            loss_dict = triplevae_loss(dx, dy, output,weights)
            loss = loss_dict["total_loss"]            
            loss.backward()
            optimizer.step()
        lr_scheduler.step(loss)

        losses.append(loss)
        recon_dX_losses.append(loss_dict["recon_dX_loss"])
        recon_dY_losses.append(loss_dict["recon_dY_loss"])
        recon_dX_s_losses.append(loss_dict["recon_dX_s_loss"])
        ortho_losses.append(loss_dict["ortho_loss"])
        equiv_losses.append(loss_dict["equiv_loss"])
    steps = range(1, len(losses) + 1)

    # only for data tranformation
    losses_np = [loss.detach().cpu().numpy() for loss in losses]
    recon_dX_losses_np = [loss_dict["recon_dX_loss"].detach().cpu().numpy() for loss_dict["recon_dX_loss"] in recon_dX_losses]
    recon_dY_losses_np = [loss_dict["recon_dY_loss"].detach().cpu().numpy() for loss_dict["recon_dY_loss"] in recon_dY_losses]
    recon_dX_s_losses_np = [loss_dict["recon_dX_s_loss"].detach().cpu().numpy() for loss_dict["recon_dX_s_loss"] in recon_dX_s_losses]
    ortho_losses_np = [loss_dict["ortho_loss"].detach().cpu().numpy() for loss_dict["ortho_loss"] in ortho_losses]
    equiv_losses_np = [loss_dict["equiv_loss"].detach().cpu().numpy() for loss_dict["equiv_loss"] in equiv_losses]
    #plt.plot(steps, losses_np, '-o',color='black')
    #plt.plot(steps, recon_dX_losses_np, '-o',color='red')
    #plt.plot(steps, recon_dY_losses_np, '-o',color='orange')
    #plt.plot(steps, recon_Y_losses_np, '-o',color='purple')
    #plt.plot(steps, recon_dX_s_losses_np, '-o',color='pink')
    #plt.plot(steps, ortho_losses_np, '-o',color='green')
    #plt.plot(steps, equiv_losses_np, '-o',color='blue')
    #plt.xlabel('step')
    #plt.ylabel('loss_value')
    #plt.title('Loss')
    #plt.legend(loc='upper right')
    #plt.show()
    return model

def evaluate_IntCIC(model, weights, dx, dy, device, return_numpy=True):
    """
    reconstruct the data and plot the reconstruction vs. the original data
    """
    
    #x = dataset.tensors[0].to(device)
    #y = dataset.tensors[1].to(device)
    dx =dx.to(device)
    dy =dy.to(device)
    model.eval().to(device)
    with torch.no_grad():
         output = model(dx,dy)
         loss_dict = triplevae_loss(dx, dy, output, weights)
         loss = loss_dict["total_loss"]
    print("loss: {:.4f}, recon_dX_loss: {:.4f}, recon_dY_loss: {:.4f}, recon_dX_s_loss: {:.4f}, ortho_loss: {:.4f}, equiv_loss: {:.4f}, kld_loss: {:.4f}, l1_loss: {:.4f}".format(  #recon_Y_s_loss: {:.4f}, recon_X_s1_loss: {:.4f},  mi_diff_Y: {:.4f}, NMI_xy: {:.4f}, MI_xy: {:.4f}, H_x: {:.4f}, H_y: {:.4f}, 
             loss.item(), loss_dict["recon_dX_loss"].item(), loss_dict["recon_dY_loss"].item(), loss_dict["recon_dX_s_loss"].item(),  #recon_X_zx_loss: {:.4f}, mi_diff_X: {:.4f}, equiv_loss: {:.4f}, 
             loss_dict["ortho_loss"].item(), loss_dict["equiv_loss"].item(), 
             loss_dict["kld_loss"].item(), loss_dict["l1_loss"].item()))
         
    recon_dx = output.recon["dx"]
    recon_dy = output.recon["dy"]
    recon_dx_s = output.recon["dx_s"]
    recon_dy_s = output.recon["dy_s"]

    recon_dx_zx = output.recon["dx_zx"]
    recon_dy_zy = output.recon["dy_zy"]
    recon_dx_s1 = output.recon["dx_s1"].detach().cpu().numpy()

    ##MSE
    MSE_X1 = nn.functional.mse_loss(dx,output.recon["dx_s"])
    MSE_X2 = nn.functional.mse_loss(dx,output.recon["dx_zx"])
    MSE_X_s,MSE_X_zx = Normalize(MSE_X1,MSE_X2)
    MSE_X3 = nn.functional.mse_loss(dy,output.recon["dy_s"])
    MSE_X4 = nn.functional.mse_loss(dy,output.recon["dy_zy"])
    MSE_dY_s,MSE_dX_zy = Normalize(MSE_X3,MSE_X4)
    MSE_X5 = nn.functional.mse_loss(dx,output.recon["dx_s1"])
    MSE_X6 = nn.functional.mse_loss(dx,output.recon["dx_zx"])
    MSE_X_s1,MSE_X_zx1 = Normalize(MSE_X5,MSE_X6)
    print("重构dX_s与原始dX的MSE:", MSE_X_s)
    print("重构dX_zx与原始dX的MSE:", MSE_X_zx)
    print("重构dY_s与原始dY的MSE:", MSE_dY_s)
    print("重构dY_zy与原始dY的MSE:", MSE_dX_zy)
    print("重构dX_s1与原始dX的MSE:", MSE_X_s1)
    print("重构dX_zx1与原始dX的MSE:", MSE_X_zx1)

    #only for GPU
    dx = dx.detach().cpu().numpy()
    dy = dy.detach().cpu().numpy()
    recon_dx = recon_dx.detach().cpu().numpy()
    recon_dx_s = recon_dx_s.detach().cpu().numpy()
    recon_dx_zx = recon_dx_zx.detach().cpu().numpy()
    recon_dy = recon_dy.detach().cpu().numpy()
    recon_dy_s = recon_dy_s.detach().cpu().numpy()
    recon_dy_zy = recon_dy_zy.detach().cpu().numpy()
    #recon_y_s = recon_y_s.detach().cpu().numpy()
    #recon_y_zy = recon_y_zy.detach().cpu().numpy()
    return output,MSE_X_s1,MSE_X_zx1

def triplevae_loss(dx, dy, output,weights):
    #1 重构损失
    recon_dX_loss = nn.functional.mse_loss(output.recon["dx"], dx)#JS_function1
    recon_dY_loss = nn.functional.mse_loss(output.recon["dy"], dy)
    recon_dX_s_loss = nn.functional.mse_loss(output.recon["dx_s"], dx)
    #3
    ortho_loss_1 = Ortho_loss(output.sample["dz_x"], output.sample["ds_x"])
    ortho_loss_2 = Ortho_loss(output.sample["dz_y"], output.sample["ds_y"])
    ortho_loss_3 = Ortho_loss(output.sample["dz_x"], output.sample["dz_y"])
    ortho_loss = ortho_loss_1 + ortho_loss_2 + ortho_loss_3 
    #4
    equiv_loss1 = nn.functional.mse_loss(output.sample["ds_x"], output.sample["ds_y"])
    equiv_loss2 = nn.functional.mse_loss(output.recon["dx_s1"], output.recon["dx_s"])#output.recon["dx"]
    equiv_loss = equiv_loss1+equiv_loss2
    #5
    l1_loss=0
    kld_loss=0
    for key in output.mean.keys():
        kld_loss += cal_kld_loss(output.mean[key], output.logvar[key])
        l1_loss += torch.mean(torch.abs(output.sample[key]))#
    total_loss = weights[0] * recon_dX_loss + weights[1] * recon_dY_loss  + weights[2] * recon_dX_s_loss+ weights[3] * ortho_loss + weights[4] * equiv_loss + weights[5] * kld_loss + weights[6] * l1_loss  
            
    #total_loss = 0.35*recon_X_loss + 0.35*recon_Y_loss + 0.13*recon_X_s_loss + 0.001*kld_loss + 0.001*l1_loss  + 0.09*ortho_loss + 0.14*equiv_loss 
    loss_dict = {"total_loss": total_loss, "recon_dX_loss": recon_dX_loss, "recon_dY_loss": recon_dY_loss, "recon_dX_s_loss": recon_dX_s_loss, #"recon_X_s1_loss": recon_X_s1_loss, 
                 #"recon_X_zx_loss": recon_X_zx_loss, #"recon_Y_s_loss": recon_Y_s_loss, "recon_Y_zy_loss": recon_Y_zy_loss, 
                 #"mi_diff_X": mi_diff_X,
                 #"NMI_xy": NMI_xy,"MI_xy": MI_xy, "H_x": H_x, "H_y": H_y, 
                 "kld_loss": kld_loss, "l1_loss": l1_loss, "ortho_loss": ortho_loss, "equiv_loss": equiv_loss}
    return loss_dict

