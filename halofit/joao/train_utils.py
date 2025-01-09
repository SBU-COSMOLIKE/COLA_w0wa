# Auxiliary functions for Power Spectrum Emulation
# Author: João Victor Silva Rebouças, May 2022
import math
import pickle
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers, optimizers
from keras.regularizers import l1_l2
from tqdm.notebook import tqdm

import torch
from torch import nn
import torch.nn.functional as F

# Bernardo's helper modules in Cython and Fortran
import pce_ as pce_utils
import ElNetFortran

#------------------------------------------------------------------------------------------------------------
# Parameter space
params = ['h', 'Omegab', 'Omegam', 'As', 'ns', 'w', 'w0pwa']
params_latex = [r'$h$', r'$\Omega_b$', r'$\Omega_m$', r'$A_s$', r'$n_s$', r'$w_0$', r'$w_0+w_a$']

# Parameter limits for training, see Table 2 from https://arxiv.org/pdf/2010.11288
lims = {}
lims['h'] = [0.61, 0.73]
lims['Omegab'] = [0.04, 0.06]
lims['Omegam'] = [0.24, 0.4]
lims['As'] = [1.7e-9, 2.5e-9]
lims['ns'] = [0.92, 1]
lims['w'] = [-1.3, -0.7]
lims['w0pwa']  = [-2.0, 0.0]
# lims['wa']  = [-0.7, 0.5]

# Reference values
ref = {}
ref['h'] = 0.67
ref['Omegab'] = 0.049
ref['Omegam'] = 0.319
ref['As'] = 2.1e-9
ref['ns'] = 0.96
ref['w'] = -1
ref['w0pwa'] = -1
# ref['wa'] = 0
params_ref = [ref[param] for param in params]

redshifts = np.linspace(3, 0, 51)

#------------------------------------------------------------------------------------------------------------

def load_set(path, w0pwa=False, load_lcdm=False):
    ks = np.loadtxt(f"{path}/ks.txt")
    lhs = np.loadtxt(f"{path}/lhs.txt")
    if w0pwa: lhs[:, -1] = lhs[:, -1] - lhs[:, -2]
    # if not w0pwa: lhs[:, -1] = lhs[:, -1] + lhs[:, -2]

    if "jonathan" in path:
        # Jonathan uses the ordering Omega_m, Omega_b, ns, As, h, w, wa
        # I use the ordering         h, Omega_b, Omega_m, As, ns, w, wa
        Omega_m, Omega_b, ns, As, h, w, wa = lhs.T
        lhs = np.column_stack((h, Omega_b, Omega_m, As, ns, w, wa))
    
    pks_lin = []
    pks_nl  = []
    num_samples = len(lhs)
    for i in range(num_samples):
        pk_lin, pk_nl = np.loadtxt(f"{path}/pk_{i}_z_0.000.txt", unpack=True)
        pks_lin.append(pk_lin)
        pks_nl.append(pk_nl)
    pks_lin = np.array(pks_lin)
    pks_nl  = np.array(pks_nl)
    
    if load_lcdm:
        pks_lin_lcdm = []
        pks_nl_lcdm = []
        for i in range(num_samples):
            pk_lin, pk_nl = np.loadtxt(f"{path}/pk_lcdm_{i}_z_0.000.txt", unpack=True)
            pks_lin_lcdm.append(pk_lin)
            pks_nl_lcdm.append(pk_nl)
        pks_lin_lcdm = np.array(pks_lin_lcdm)
        pks_nl_lcdm  = np.array(pks_nl_lcdm)

    if load_lcdm:
        return lhs, ks, pks_lin, pks_nl, pks_lin_lcdm, pks_nl_lcdm
    else:
        return lhs, ks, pks_lin, pks_nl

class HalofitSet:
    def __init__(self, path, use_boost_ratio=False):
        self.num_pcs = None
        self.use_boost_ratio = use_boost_ratio
        data = load_set(path, w0pwa="w0pwa" in path, load_lcdm=use_boost_ratio)
        if not use_boost_ratio: self.lhs, self.ks, self.pks_lin, self.pks_nl = data
        else: self.lhs, self.ks, self.pks_lin, self.pks_nl, self.pks_lin_lcdm, self.pks_nl_lcdm = data
        self.boosts = self.pks_nl/self.pks_lin
        self.logboosts = np.log(self.boosts)
        if use_boost_ratio:
            self.boosts_lcdm = self.pks_nl_lcdm/self.pks_lin_lcdm
            self.boost_ratio = self.boosts/self.boosts_lcdm
            self.logboost_ratio = np.log(self.boost_ratio)

    def change_ks(self, ks):
        boosts = []
        for boost in self.boosts:
            boost_interp = CubicSpline(self.ks, boost)
            new_boost = boost_interp(ks)
            boosts.append(new_boost)
        self.boosts = boosts
        self.logboosts = np.log(self.boosts)
        self.ks = ks
        if self.num_pcs is not None: self.prepare(self.num_pcs)

    def update(self, cosmos, boosts):
        self.lhs = np.vstack([self.lhs, cosmos])
        self.boosts = np.vstack([self.boosts, boosts])
        self.logboosts = np.log(self.boosts)
        if self.num_pcs is not None: self.prepare(self.num_pcs)

    def prepare(self, num_pcs):
        self.num_pcs = num_pcs
        self.param_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(self.lhs)
        self.lhs_norm = self.param_scaler.transform(self.lhs)
        self.boost_scaler = Scaler()
        if not self.use_boost_ratio:
            self.boost_scaler.fit(self.logboosts)
            self.logboosts_norm = self.boost_scaler.transform(self.logboosts)    
        else:
            self.boost_scaler.fit(self.logboost_ratio)
            self.logboosts_norm = self.boost_scaler.transform(self.logboost_ratio)
        self.num_pcs = num_pcs
        self.pca = PCA(n_components=num_pcs)
        self.pcs = self.pca.fit_transform(self.logboosts_norm)

    def plot_boosts(self):
        for boost in self.boosts:
            plt.semilogx(self.ks, boost)
        plt.title(f"Halofit train boosts for z = 0")
        plt.xlabel("k")
        plt.ylabel("Boost")
    
    def plot_logboosts(self):
        for boost in self.logboosts:
            plt.semilogx(self.ks, boost)
        plt.title(f"Halofit train boosts for z = 0")
        plt.xlabel("k")
        plt.ylabel("Log(Boost)")

    def plot_logboosts_norm(self):
        for boost in self.logboosts_norm:
            plt.semilogx(self.ks, boost)
        plt.title(f"Halofit train boosts for z = 0")
        plt.xlabel("k")
        plt.ylabel("Boost")

    def plot_lhs(self, model_to_map_errors=None):
        D = 7
        assert D == len(self.lhs[0])
        if model_to_map_errors is not None:
            model_predictions = model_to_map_errors.predict(self.lhs)
            errors = np.abs(model_predictions/self.boosts - 1)
            max_errors = []
            for error in errors: max_errors.append(np.log10(np.amax(error)))
        fig, axs = plt.subplots(D, D, figsize=(15, 15), gridspec_kw={"hspace": 0.1, "wspace": 0.1})
        for row in range(D):
            for col in range(D):
                ax = axs[row, col]
                if row == D-1: ax.set_xlabel(params_latex[col])
                else: ax.set_xticks([])
                if col == 0: ax.set_ylabel(params_latex[row])
                else: ax.set_yticks([])
                if col >= row: ax.remove()
                x_index = params.index(params[col])
                y_index = params.index(params[row])
                x_params = [sample[x_index] for sample in self.lhs]
                y_params = [sample[y_index] for sample in self.lhs]
                if model_to_map_errors is not None: im = ax.scatter(x_params, y_params, s=6, alpha=0.75, c=max_errors)
                else: im = ax.scatter(x_params, y_params, s=2, alpha=0.5)
        
        # See https://stackoverflow.com/questions/13784201/how-to-have-one-colorbar-for-all-subplots
        if model_to_map_errors is not None:
            fig.subplots_adjust(right=0.9)
            cbar_ax = fig.add_axes([0.75, 0.25, 0.025, 0.35])
            fig.colorbar(im, cax=cbar_ax, shrink=0.75, label="Log Error")
        return fig
        
class COLAModel():
    def __init__(self, trainSet):
        self.boost_scaler = trainSet.boost_scaler
        self.param_scaler = trainSet.param_scaler
        self.pca = trainSet.pca

    def fit(self, trainSet, num_epochs):
        raise NotImplementedError("ERROR: COLAModel must override `fit` method")

    def predict_pcs(self, x):
        raise NotImplementedError("ERROR: COLAModel must override `predict_pcs` method")

    def predict(self, x):
        pcs = self.predict_pcs(x)
        logboost_norm = self.pca.inverse_transform(pcs)
        logboost = self.boost_scaler.inverse_transform(logboost_norm)
        return np.exp(logboost)

    def plot_errors(self, testSet):
        preds = self.predict(testSet.lhs)
        fig, ax = plt.subplots()
        targets = testSet.boosts if not testSet.use_boost_ratio else testSet.boost_ratio
        for pred, target in zip(preds, targets):
            error = pred/target - 1
            ax.semilogx(testSet.ks, error)
        ax.fill_between(testSet.ks, -0.0025, 0.0025, color="gray", alpha=0.75)
        ax.fill_between(testSet.ks, -0.005, 0.005, color="gray", alpha=0.5)
        return fig, ax

    def get_outliers(self, testSet, log=False):
        boosts_pred = self.predict(testSet.lhs)
        cosmos = []
        boosts = []
        for boost_test, boost_pred, cosmo in zip(boosts_pred, testSet.boosts, testSet.lhs):
            error = np.abs(boost_pred/boost_test - 1)
            if np.any(error > 0.005):
                h, Omega_b, Omega_m, As, ns, w, w0pwa = cosmo
                if log: print(f"Outlier: [{h=}, {Omega_b=}, {Omega_m=}, {As=}, {ns=}, {w=}, {w0pwa=}] has error {np.amax(error)} ")
                cosmos.append(cosmo)
                boosts.append(boost_test)
        return cosmos, boosts

    def save(self, path):
        with open(path, "wb") as f: pickle.dump(self, f)

def load_model(path):
    with open(path, "rb") as f: model = pickle.load(f)
    return model

class COLA_NN_Keras(COLAModel):
    def __init__(self, trainSet, num_layers=3, num_neurons=2048):
        super().__init__(trainSet)

        self.mlp = generate_mlp(
            input_shape=len(trainSet.lhs[0]),
            output_shape=len(trainSet.pcs[0]),
            num_layers=num_layers,
            num_neurons=num_neurons,
            activation="custom",
            alpha=0,
            l1_ratio=0
        )
    
    def fit(self, trainSet, num_epochs, decayevery, decayrate):
        try: nn_model_train_keras(self.mlp, epochs=num_epochs, input_data=trainSet.lhs_norm, truths=trainSet.pcs, decayevery=decayevery, decayrate=decayrate)
        except KeyboardInterrupt:
            print("Training interrupted.")
            return
    
    def predict_pcs(self, x):
        x_norm = self.param_scaler.transform(x)
        return self.mlp(x_norm)

class COLA_NN_Torch(COLAModel):
    def __init__(self, trainSet, num_neurons, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__(trainSet)
        self.mlp = Net(len(trainSet.lhs[0]), len(trainSet.pcs[0]), num_neurons).to(device)

    def fit(self, num_epochs, trainSet, testSet, decayevery, decayrate):
        try:
            train_model_torch(self.mlp, trainSet, testSet, num_epochs, decayevery, decayrate)
        except KeyboardInterrupt:
            print("Training interrupted.")
            return

    def predict_pcs(self, x, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        x_norm = torch.Tensor(self.param_scaler.transform(x)).to(device)
        pcs = self.mlp(x_norm).cpu().detach().numpy()
        return pcs

class COLA_PCE(COLAModel):
    def __init__(self, trainSet, max_order):
        super().__init__(trainSet)
        d = len(trainSet.lhs[0])
        max_values = np.array([max_order for _ in range(d)]) # FOI ESSE melhor com tudo 5
        numbers = np.arange(max_order + 1)
        combinations = np.array(list(product(numbers, repeat=d)), dtype=np.float64)
        filtered_data = pce_utils.filter_q0_norm(combinations, threshold=5)
        self.allowcomb = pce_utils.filter_combinations(filtered_data/max_values,  p=0.95 + 1e-12, threshold=1 + 1e-12) #best=6
        self.allowcomb = self.allowcomb*max_values
        X = pce_utils.calculate_power_expansion(self.allowcomb, np.clip(trainSet.lhs_norm, -1, 1))
        self.n_samples, self.n_features = X.shape
        self.n_targets = trainSet.pcs.shape[1]
        self.n_tasks = trainSet.pcs.shape[1]
        W = np.asfortranarray(
            np.zeros(
                (self.n_targets, self.n_features),
                dtype=X.dtype.type,
                order="F"
            )
        ) # W are the coefficients of the polynomial expansion
        
        R = np.zeros((self.n_samples, self.n_tasks), dtype=X.dtype.type, order='F')
        norm_cols_X = np.zeros(self.n_features, dtype=X.dtype.type)
        norm_cols_X = (np.asarray(X)**2).sum(axis=0)

        R = trainSet.pcs - np.dot(X, W.T)

        self.W = np.asfortranarray(W)
        self.X = np.asfortranarray(X) 
        self.R = np.asfortranarray(R) 
        self.norm_cols_X = np.asfortranarray(norm_cols_X)

    def fit(self, trainSet, num_epochs, alpha=1e-5, l1_ratio=0.05):
        # Bernardo recommended:
        # z = 0: alpha = 1e-5, l1_ratio = 0.05
        # else: alpha = 1e-6, l1_ratio = 0.9
        self.l1_reg =  alpha *  l1_ratio * self.n_samples
        self.l2_reg =  alpha * (1.0 -  l1_ratio) * self.n_samples
        for _ in range(num_epochs):
            ElNetFortran.fit(
                self.n_features,
                1,
                self.l1_reg,
                self.l2_reg,
                self.W,
                self.X,
                self.R,
                self.norm_cols_X
            )

    def predict_pcs(self, x):
        x_norm = self.param_scaler.transform(x)
        yteste = pce_utils.calculate_power_expansion(self.allowcomb, np.clip(x_norm, -1, 1))
        pcs_pred = yteste@self.W.T
        return pcs_pred

class COLA_NPCE(COLA_PCE):
    def __init__(self, trainSet, max_order, num_neurons, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__(trainSet, max_order)
        self.mlp = Net(len(trainSet.pcs[0]), len(trainSet.pcs[0]), num_neurons).to(device)

    def fit(self, trainSet, num_epochs_pce, alpha, l1_ratio, num_epochs_nn, testSet, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        print("Training PCE.")
        super().fit(trainSet, num_epochs_pce, alpha, l1_ratio)
        print("Finished PCE training, starting NN training.")
        pcs_pred = super().predict_pcs(trainSet.lhs)
        pcs_pred_test = super().predict_pcs(testSet.lhs)
        optimizer = torch.optim.Adam(self.mlp.parameters(), lr=0.01)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1500, gamma=0.9)
        loss_fn = torch.nn.MSELoss()
        inputs = torch.tensor(pcs_pred, dtype=torch.float64).to(device)
        truths = torch.tensor(trainSet.pcs, dtype=torch.float64).to(device)

        def closure():
            optimizer.zero_grad()
            outputs = self.mlp(inputs).to(device)
            loss = loss_fn(outputs, truths)
            loss.backward()
            return loss

        loss_treshold = 1e-6
        test_loss = 0
        for epoch in tqdm(range(num_epochs_nn)):
            try:
                train_loss = optimizer.step(closure).item()
                lr_scheduler.step()
                predictions = self.mlp(torch.Tensor(pcs_pred_test).to(device)).cpu().detach().numpy() # PCs
                logboosts_norm_pred = trainSet.pca.inverse_transform(predictions)
                logboosts_pred = trainSet.boost_scaler.inverse_transform(logboosts_norm_pred)
                test_loss = mean_squared_error(testSet.boosts, np.exp(logboosts_pred))
                print(f'Epoch {epoch} => Loss = {train_loss:.4g}, Test_Loss: {test_loss:.4g}', end="\r")
                if test_loss < loss_treshold:
                    break
            
            except KeyboardInterrupt:
                print("Training interrupted.")
                break

    def predict_pcs(self, x, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        pcs_pce = super().predict_pcs(x)
        pcs = self.mlp.forward(torch.Tensor(pcs_pce).to(device)).cpu().detach().numpy()
        return pcs

class COLA_KAN(COLAModel):
    def __init__(self, trainSet, num_neurons, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__(trainSet)
        self.kan = customKAN(num_neurons, input_dim=len(trainSet.lhs[0]), output_dim=len(trainSet.pcs[0])).to(device)

    def fit(self, trainSet, num_epochs, testSet, decayevery, decayrate):
        try:
            train_model_torch(self.kan, trainSet, testSet, num_epochs, decayevery=decayevery, decayrate=decayrate)
        except KeyboardInterrupt:
            print("Training interrupted.")
            return

    def predict_pcs(self, x, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        x_norm = torch.Tensor(self.param_scaler.transform(x)).to(device)
        return self.kan(x_norm).cpu().detach().numpy()


#------------------------------------------------------------------------------------------------------------

# Bernardo's code for KAN model in PyTorch
class Scaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

    def transform(self, X):
        return (X - self.mean) / self.std

    def inverse_transform(self, X):
        return (X* self.std) + self.mean

class CustomActivationLayerPyTorch(nn.Module):
    def __init__(self, units):
        super(CustomActivationLayerPyTorch, self).__init__()
        self.units = units

        self.beta = nn.Parameter(torch.randn(self.units))
        self.gamma = nn.Parameter(torch.randn(self.units))

    def forward(self, x):
        # Implementando a função conforme descrita no TensorFlow
        return self.gamma* x + torch.sigmoid(self.beta * x) * (1 - self.gamma)* x

class Net(nn.Module):
    def __init__(self, input_dim, output_dim, num_neurons):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(input_dim, num_neurons)
        self.custom_layer = CustomActivationLayerPyTorch(num_neurons)#
        self.layer2 = nn.Linear(num_neurons, num_neurons)
        self.custom_layer2 = CustomActivationLayerPyTorch(num_neurons)
        self.layer3 = nn.Linear(num_neurons, num_neurons)
        self.custom_layer3 = CustomActivationLayerPyTorch(num_neurons)     
        self.layer4 = nn.Linear(num_neurons, output_dim)
        # self.custom_layer4 = CustomActivationLayerPyTorch(output_dim)     

    def forward(self, x):
        x = self.layer1(x)
        x = self.custom_layer(x)
        x = self.layer2(x)
        x = self.custom_layer2(x)
        x = self.layer3(x)
        x = self.custom_layer3(x)
        x = self.layer4(x)
        # x = self.custom_layer4(x)
        return x

class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output
        
        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )

class customKAN(nn.Module):
    def __init__(self, num_neurons, input_dim, output_dim):
        super(customKAN, self).__init__()
         
        self.chebykan1 = KANLinear(input_dim,
            num_neurons,
            grid_size=3,
            spline_order=15,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            enable_standalone_scale_spline=True,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1]
        )
        # self.custom_layer1 = CustomActivationLayerPyTorch(num_neurons)
        
        self.chebykan2 = KANLinear(num_neurons,
            output_dim,
            grid_size=3,
            spline_order=15,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            enable_standalone_scale_spline=True,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1]
        )
        # self.custom_layer2 = CustomActivationLayerPyTorch(num_neurons)#
        
    def forward(self, x):
        # ident = x
        # x = nn.Tanh()(x)
        x = self.chebykan1(x) 
        # x = self.custom_layer1(x)
        x = self.chebykan2(x)
        return x # + ident

def train_model_torch(model, trainSet, testSet, num_epochs=6_000, decayevery=1000, decayrate=0.95, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decayevery, gamma=decayrate)
    loss_fn = torch.nn.MSELoss()
    inputs = torch.tensor(trainSet.lhs_norm, dtype=torch.float64).to(device)
    truths = torch.tensor(trainSet.pcs, dtype=torch.float64).to(device)
    test_inputs = torch.tensor(trainSet.param_scaler.transform(testSet.lhs), dtype=torch.float64).to(device)
    
    def closure():
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, truths)
        loss.backward()
        return loss

    loss_treshold = 1e-6
    test_loss = 0

    for epoch in tqdm(range(num_epochs)):
        try:
            train_loss = optimizer.step(closure).item()
            lr_scheduler.step()
            
            predictions = model(torch.Tensor(test_inputs).to(device)).cpu().detach().numpy() # PCs
            logboosts_norm_pred = trainSet.pca.inverse_transform(predictions)
            logboosts_pred = trainSet.boost_scaler.inverse_transform(logboosts_norm_pred)
            test_loss = mean_squared_error(testSet.boosts, np.exp(logboosts_pred))

            print(f'Epoch {epoch} => Loss = {train_loss:.4g}, Test_Loss: {test_loss:.4g}', end="\r")

            if test_loss < loss_treshold:
                break
        
        except KeyboardInterrupt:
            print("Training interrupted.")
            break

#------------------------------------------------------------------------------------------------------------

# Joao's code for NN in Keras, used in the COLA_NN_Keras class

class CustomActivationLayer(layers.Layer):
    def __init__(self, units, **kwargs):
        super(CustomActivationLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.beta = self.add_weight(shape=(self.units,), initializer='random_normal', trainable=True, name="beta")
        self.gamma = self.add_weight(shape=(self.units,), initializer='random_normal', trainable=True, name="gamma")
        super(CustomActivationLayer, self).build(input_shape)

    def call(self, x):
        # See e.g. https://arxiv.org/pdf/1911.11778.pdf, Equation (8)
        func = tf.add(self.gamma, tf.multiply(tf.sigmoid(tf.multiply(self.beta, x)), tf.subtract(1.0, self.gamma)))
        return tf.multiply(func, x)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

def generate_mlp(input_shape, output_shape, num_layers, num_neurons, activation="custom", alpha=0.01, l1_ratio=0.01, learning_rate=1e-3, optimizer='adam', loss='mse'):
    '''
    Generates an MLP model with `num_res_blocks` residual blocks.
    '''
    reg = l1_l2(l1=alpha*l1_ratio, l2=alpha*(1-l1_ratio)) if alpha != 0 else None
    
    # Define the input layer
    inputs = layers.Input(shape=(input_shape,))
    
    # Define the first hidden layer separately because it needs to connect with the input layer
    x = layers.Dense(num_neurons, kernel_regularizer=reg)(inputs)
    if activation == "custom":
        x = CustomActivationLayer(num_neurons)(x)
    elif activation == "relu":
        x = keras.activations.relu(x)
    elif activation == "sigmoid":
        x = keras.activations.sigmoid(x)
    else:
        raise Exception(f"Unexpected activation {activation}")
    
    # Add more hidden layers
    for _ in range(num_layers - 1): # subtract 1 because we've already added the first hidden layer
        x = layers.Dense(num_neurons, kernel_regularizer=reg)(x)
        if activation == "custom":
            x = CustomActivationLayer(num_neurons)(x)
        elif activation == "relu":
            x = keras.activations.relu(x)
        elif activation == "sigmoid":
            x = keras.activations.sigmoid(x)
        else:
            raise Exception(f"Unexpected activation {activation}")

    # Define the output layer
    outputs = layers.Dense(output_shape)(x)
    
    # Choose the optimizer
    if optimizer.lower() == 'adam':
        opt = optimizers.Adam(learning_rate=learning_rate)
    elif optimizer.lower() == 'sgd':
        opt = optimizers.SGD(learning_rate=learning_rate, momentum=0.99, nesterov=True)
    else:
        raise ValueError(f"Unhandled optimizer: {optimizer}")

    # Construct and compile the model
    model = models.Model(inputs=inputs, outputs=outputs)
    model.summary()
    # Compile the model
    model.compile(optimizer=opt, loss=loss)

    return model

#------------------------------------------------------------------------------------------------------------

def generate_resnet(input_shape, output_shape, num_res_blocks=1, num_of_neurons=512, activation="relu", alpha=1e-5, l1_ratio=0.1, dropout=0.1):
    '''
    Generates a ResNet model with `num_res_blocks` residual blocks.
    '''
    nn_layers = []
    regularization_term = l1_l2(l1=alpha*l1_ratio, l2=alpha*(1-l1_ratio))
    
    # Adding layers
    input_layer = layers.Input(shape=input_shape)
    
    # Adding first residual block
    hid1 = layers.Dense(units=num_of_neurons,
         kernel_regularizer=regularization_term,
         bias_regularizer=regularization_term)(input_layer)
    act1 = CustomActivationLayer(num_of_neurons)(hid1)
    
    hid2 = layers.Dense(units=num_of_neurons,
         kernel_regularizer=regularization_term,
         bias_regularizer=regularization_term)(act1)
    act2 = CustomActivationLayer(num_of_neurons)(hid2)
    residual = layers.Add()([act1, act2])
    
    if num_res_blocks > 1:
        for i in range(num_res_blocks - 1):
            hid1 = layers.Dense(units=num_of_neurons,
                 kernel_regularizer=regularization_term,
                 bias_regularizer=regularization_term)(residual)
            act1 = CustomActivationLayer(num_of_neurons)(hid1)
            hid2 = layers.Dense(units=num_of_neurons,
                 kernel_regularizer=regularization_term,
                 bias_regularizer=regularization_term)(act1)
            act2 = CustomActivationLayer(num_of_neurons)(hid2)
            residual = layers.Add()([act1, act2])
    
    output_layer = layers.Dense(units=output_shape)(residual)
    
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    
    model.summary()
    
    model.compile(
        optimizer = keras.optimizers.Adam(),
        loss = keras.losses.MeanAbsoluteError()
    )
    
    return model

#------------------------------------------------------------------------------------------------------------

def nn_model_train_keras(model, epochs, input_data, truths, validation_features=None, validation_truths=None, decayevery=None, decayrate=None):
    '''
    Trains a neural network model that emulates the truths from the input_data
    Can program the number of epochs and a step-based learning rate decay
    '''
    # See https://stackoverflow.com/questions/44931689/how-to-disable-printing-reports-after-each-epoch-in-keras
    class PrintCallback(tf.keras.callbacks.Callback):
        SHOW_NUMBER = 10
        epoch = 0

        def on_epoch_begin(self, epoch, logs=None):
            self.epoch = epoch

        def on_epoch_end(self, batch, logs=None):
            print(f'Epoch: {self.epoch} => Loss = {logs["loss"]}', end="\r")
    
    if decayevery and decayrate: 
        def scheduler(epoch, learning_rate):
            # Halves the learning rate at some points during training
            if epoch != 0 and epoch % decayevery == 0:
                return learning_rate/decayrate
            else:
                return learning_rate
        learning_scheduler = keras.callbacks.LearningRateScheduler(scheduler)
    else:
        learning_scheduler = keras.callbacks.LearningRateScheduler(lambda epoch, learning_rate: learning_rate)
    
    if validation_features and validation_truths:
        history = model.fit(
            input_data,
            truths,
            batch_size = 30,
            epochs = epochs,
            validation_data = (validation_features, validation_truths),
            callbacks=[learning_scheduler, PrintCallback()],
            verbose=0
        )
    else:
        history = model.fit(
            input_data,
            truths,
            batch_size = 30,
            epochs = epochs,
            callbacks=[learning_scheduler, PrintCallback()],
            verbose=0
        )
    
    last_loss = history.history['loss'][-1]
    return last_loss

#------------------------------------------------------------------------------------------------------------