"""
    Utility for creating training and test sets of the w0wa model using the LHS design
"""
import argparse
import sys
import os
import numpy as np
from scipy.stats.qmc import LatinHypercube, scale
import camb
from camb import model

# Parameter space
params = ['h', 'Omegab', 'Omegam', 'As', 'ns', 'w', 'w0pwa']

# EE2 parameter limits, see Table 2 from https://arxiv.org/pdf/2010.11288
lims = {}
lims['h']      = [0.61, 0.73]
lims['Omegab'] = [0.04, 0.06]
lims['Omegam'] = [0.24, 0.4]
lims['As']     = [1.7e-9, 2.5e-9]
lims['ns']     = [0.92, 1]
lims['w']      = [-1.3, -0.7]
lims['w0pwa']  = [-2.0, -0.2]
lims['wa']     = [-0.7, 0.5]

# Stretching the training box
lims_stretch = {}
for param in params:
    param_min, param_max = lims[param]
    length = param_max - param_min
    factor = 0.1 # New length = (1 + factor)*original_length
    lims_stretch[param] = [param_min - factor*length/2, param_max + factor*length/2]

# Fixing w0pwa
lims_stretch['w0pwa'] = [-2.2, 0.0]
lims_stretch['wa'] = [-0.76, 0.56]

redshifts = np.linspace(3, 0, 10)

# Note: Ref cosmology doesn't provide tau, using same as Guilherme
def get_pk(h, Omega_b, Omega_c, As, ns, w0, wa, redshifts=[0], tau=0.078, nonlinear=True):
    '''
    Returns a list [k, Pk_lin, Pk_nl], the scales and power spectrum for the given cosmology.
    Uses 200 k-bins from k = 1e-2 to k = pi (h/Mpc).
    '''
    n_points = 200
    k_min = 1e-2
    k_max = np.pi
    cosmology = camb.set_params(# Background
                                H0 = 100*h, ombh2=Omega_b*h**2, omch2=Omega_c*h**2,
                                TCMB = 2.7255,
                                # Dark Energy
                                dark_energy_model='ppf', w=w0, wa=wa,
                                # Neutrinos
                                nnu=3.046, mnu=0.058, num_nu_massless=0, num_massive_neutrinos=3,
                                # Initial Power Spectrum
                                As=As, ns=ns, tau=tau,
                                YHe = 0.246, WantTransfer=True)
    cosmology.set_matter_power(redshifts=redshifts, kmax=20.0)
    cosmology.NonLinear = model.NonLinear_none
    results = camb.get_results(cosmology)
    
    # Calculating Linear Pk
    ks, _, pk_lin = results.get_matter_power_spectrum(minkh=k_min, maxkh=k_max, npoints=n_points)
    
    # Recalculating Pk with Nonlinear
    cosmology.NonLinear = model.NonLinear_both
    cosmology.NonLinearModel.set_params(halofit_version='takahashi')
    results.calc_power_spectra(cosmology)
    ks, _, pk_nl = results.get_matter_power_spectrum(minkh=k_min, maxkh=k_max, npoints=n_points)
    return ks, pk_lin, pk_nl

argp = argparse.ArgumentParser(
    prog="Halofit Dataset Generator",
    description="A Python script that generates Halofit Boost datasets with the w0wa model"
)
argp.add_argument("num", type=int, help="Number of points in the LHS")
argp.add_argument("path", help="Path to store the boosts, LHS, ks and zs")
argp.add_argument("--stretch", action="store_true", help="With this flag, stretch the LHS by 10 percent for training sets")
argp.add_argument("--project", action="store_true", help="With this flag, the LHS is projected in the LCDM subspace")
argp.add_argument("--use_wa", action="store_true", help="With this flag, the samplng occurs over wa instead of w0+wa")

if __name__ == "__main__":
    print("----- LHS generator for w0wa parameter space -----")
    args = argp.parse_args()

    if args.use_wa:
        print("[INFO] Using wa")
        params.pop()
        params.append("wa")
    else:
        print("[INFO] Using w0+wa")
    
    if not args.project:
        print(f"[INFO] Making LHS of length {args.num}, saving power spectra in {args.path}")
        sampler = LatinHypercube(d=len(params))
        lhs_normalized = sampler.random(n=args.num)
        if not args.stretch:
            lower_bounds = [lims[param][0] for param in params]
            upper_bounds = [lims[param][1] for param in params]
        else:
            lower_bounds = [lims_stretch[param][0] for param in params]
            upper_bounds = [lims_stretch[param][1] for param in params]
        lhs = np.array(scale(lhs_normalized, lower_bounds, upper_bounds))
        os.mkdir(args.path)
        np.savetxt(f"{args.path}/lhs.txt", lhs, header=" ".join(params))
        print(f"[INFO] Saved lhs in {args.path}/lhs.txt")
        np.savetxt(f"{args.path}/redshifts.txt", redshifts, fmt="%.2f")
        print(f"[INFO] Saved redshifts in {args.path}/redshifts.txt")
        for i, sample in enumerate(lhs):
            h, Omega_b, Omega_m, As, ns, w, wa_or_w0pwa = sample
            Omega_c = Omega_m - Omega_b - 0.058/(94.13*h*h)
            if args.use_wa: wa = wa_or_w0pwa
            else: wa = wa_or_w0pwa - w
            k, pk_lin, pk_nl = get_pk(h, Omega_b, Omega_c, As, ns, w, wa)
            for z, pk_lin_at_z, pk_nl_at_z in zip(reversed(redshifts), pk_lin, pk_nl):
                np.savetxt(f"{args.path}/pk_{i}_z_{z:.3f}.txt", np.column_stack((pk_lin_at_z, pk_nl_at_z)))
        print(f"[INFO] Saved linear and nonlinear pk in {args.path}/")

        np.savetxt(f"{args.path}/ks.txt", k)
        print(f"[INFO] Saved ks in {args.path}/ks.txt")

    else:
        print(f"[INFO] Making LCDM projections of lhs from {args.path}/lhs.txt")
        lhs = np.loadtxt(f"{args.path}/lhs.txt")
        for i, sample in enumerate(lhs):
            h, Omega_b, Omega_m, As, ns, w, w0pwa = sample
            Omega_c = Omega_m - Omega_b - 0.058/(94.13*h*h)
            k, pk_lin, pk_nl = get_pk(h, Omega_b, Omega_c, As, ns, -1, 0)
            for z, pk_lin_at_z, pk_nl_at_z in zip(reversed(redshifts), pk_lin, pk_nl):
                np.savetxt(f"{args.path}/pk_lcdm_{i}_z_{z:.3f}.txt", np.column_stack((pk_lin_at_z, pk_nl_at_z)))
        print(f"[INFO] Saved linear and nonlinear pk in {args.path}/")
    print("----- Finished! -----")
