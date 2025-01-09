import numpy as np

def load_set(path):
    ks = np.loadtxt(f"{path}/ks.txt")
    zs = np.loadtxt(f"{path}/redshifts.txt")
    lhs = np.loadtxt(f"{path}/lhs.txt")
    num_samples = len(lhs)

    pks_lin_at_z = np.zeros((num_samples, len(ks)))
    pks_nl_at_z  = np.zeros((num_samples, len(ks)))

    for i in range(num_samples):
        pks_lin_at_z[i] = np.loadtxt(f"{path}/pk_lin/pk_lin_{i}_z0.00.txt")
        pks_nl_at_z[i]  = np.loadtxt(f"{path}/pk_nl/pk_nl_{i}_z0.00.txt")

    return lhs, ks, pks_lin_at_z, pks_nl_at_z