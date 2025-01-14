"""
    Transfer Function Generator for w0wa cosmologies
    Author: João Rebouças, Oct 2024
    Usage: do `python3 transfer_function_generator.py --help`
    In case of any bugs, contact João ASAP!
"""

import argparse
import os
import time
import numpy as np
from classy import Class
from scipy.interpolate import interp1d

# COLA integration nodes
zs_COLA = np.flip([
    0.000, 0.020, 0.041, 0.062, 0.085, 0.109, 0.133, 0.159, 0.186, 0.214, 0.244, 0.275, 0.308, 
    0.342, 0.378, 0.417, 0.457, 0.500, 0.543, 0.588, 0.636, 0.688, 0.742, 0.800, 0.862, 0.929, 
    1.000, 1.087, 1.182, 1.286, 1.400, 1.526, 1.667, 1.824, 2.000, 2.158, 2.333, 2.529, 2.750, 
    3.000, 3.286, 3.615, 4.000, 4.455, 5.000, 5.667, 6.500, 7.571, 9.000, 11.000, 14.000, 19.000, 
    19.500, 20.000
])

def generate_transfers(Omega_m, Omega_b, h, As, ns, w, wa, path="./transfers/", sim_number=0):
    """
        Main function that generates transfer functions in the N-body gauge for w0wa cosmologies using CLASS
        The transfers are saved inside `path` in files named 'data_transfer_z0.000.txt'...
        A 'transferinfo.txt' file - the file that needs to be pointed to COLA - is also generated inside `path`
    """
    m_nu = 0.058 # eV
    Omega_nu = (m_nu/94.13)/(h*h) # See Dodelson 2nd edition, equation 2.84
    Omega_c = Omega_m - Omega_b - Omega_nu
    k_per_decade = 25 # This parameter controls the k-resolution of the transfer functions
    z_max_pk = 250.0  # This parameter might need to be increased for weird cosmologies, but generally 250 should be fine

    cosmo_compute = Class()
    cosmo_compute.set({
        # General settings
        'output': 'dTk, vTk',
        'nbody_gauge_transfer_functions': 'yes',
        'format': 'camb',
        'h': h,
        'Omega_b': f'{Omega_b}',
        'Omega_cdm': Omega_c,
        'A_s': As,
        'n_s': ns,
        'w0_fld': f'{w}',
        'wa_fld': f'{wa}',
        'Omega_Lambda': 0, # Needs to be set to zero so CLASS uses fluid
        'use_ppf': 'yes',
        'radiation_streaming_approximation': 3, # Turn off radiation approximation
        'ur_fluid_approximation': 2, # Turn off massless nu fluid approximation
        'N_ur': 0, # Number of masless neutrinos
        'N_ncdm': 1, # Number of massive neutrino species
        'm_ncdm': m_nu/3, # Mass for each neutrino species
        'deg_ncdm': 3.0, # Eigenstate degeneracy
        'ncdm_fluid_approximation': 3,
        # Important precision settings
        'z_max_pk': z_max_pk,
        'evolver': 0, # 0 = runge kutta solver 
        'k_per_decade_for_pk': k_per_decade,
        'P_k_max_h/Mpc' : 10, # NOTE: Needs to be higher than the Nyquist frequency of the simulations
        'l_max_ncdm': 500,    # NOTE: increasing Boltzmann hierarchy moments slows down generation, but is important for simulations
        'l_max_g': 3500,      # NOTE: increasing Boltzmann hierarchy moments slows down generation, but is important for simulations
        'l_max_pol_g': 3500,  # NOTE: increasing Boltzmann hierarchy moments slows down generation, but is important for simulations
        'l_max_ur': 3500,     # NOTE: increasing Boltzmann hierarchy moments slows down generation, but is important for simulations
        # Precision settings end
        'background_verbose': 0, 
        'perturbations_verbose': 0,
        'gauge': 'synchronous',
    })
    cosmo_compute.compute()
    
    background = cosmo_compute.get_background()

    background_tau = background['conf. time [Mpc]'] # Read conformal times in background table
    background_z = background['z']                  # Read redshift
    background_a = 1./(1.+background_z)
    background_H = background['H [1/Mpc]']

    tau_num = 1000
    tau = np.linspace(1000, background_tau[-1], tau_num)

    background_a_at_tau = interp1d(background_tau, background_a, fill_value="extrapolate")
    background_z_at_tau = interp1d(background_tau, background_z, fill_value="extrapolate")
    background_tau_at_z = interp1d(background_z, background_tau, fill_value="extrapolate")

    max_z_needed = background_z_at_tau(tau[0])
    if max_z_needed > z_max_pk:
        print(f'ERROR: you must increase the value of z_max_pk to at least {max_z_needed}')
        exit(1)
    
    # Get transfer functions at each time and build arrays Theta0(tau,k) and phi(tau,k)
    for i in range(tau_num):
        z = background_z_at_tau(tau[i])
        if z < 0: z = 0 # NOTE: the interpolator may give a value of z < 0 for today, tau_0, and CLASS complains about that
        one_time = cosmo_compute.get_transfer(z) # transfer functions at each time tau
        if i == 0:  # if this is the first time in the loop: create the arrays (k, Theta0, phi)
            k = one_time['k (h/Mpc)']*cosmo_compute.h()
            k_num = len(k)
            d_GR_CAMB = np.zeros((tau_num,k_num))
            d_m_Nb_CAMB = np.zeros((tau_num,k_num))
            d_cdm_Nb_CAMB = np.zeros((tau_num,k_num))
            d_b_Nb_CAMB = np.zeros((tau_num,k_num))
            d_g_Nb_CAMB = np.zeros((tau_num,k_num))
            d_ur_Nb_CAMB = np.zeros((tau_num,k_num))
            d_total_Nb_CAMB = np.zeros((tau_num,k_num))
            d_total_DE_Nb_CAMB = np.zeros((tau_num,k_num))

        camb_format_factor = -1/k**2
        d_m_Nb_CAMB[i,:] = one_time['d_m'][:]*camb_format_factor
        d_cdm_Nb_CAMB[i,:] = one_time['d_cdm'][:]*camb_format_factor
        d_b_Nb_CAMB[i,:] = one_time['d_b'][:]*camb_format_factor
        d_g_Nb_CAMB[i,:] = one_time['d_g'][:]*camb_format_factor
        d_ur_Nb_CAMB[i,:] = one_time['d_ncdm[0]'][:]*camb_format_factor
        d_total_Nb_CAMB[i,:] = one_time['d_tot'][:]*camb_format_factor
        d_total_DE_Nb_CAMB[i,:] = one_time['d_fld'][:]*camb_format_factor

    ks_COLA = one_time['k (h/Mpc)']

    T_d_GR_COLA = np.zeros((len(zs_COLA),len(ks_COLA)),'float64') 
    T_d_m_Nb_COLA = np.zeros((len(zs_COLA),len(ks_COLA)),'float64')
    T_d_cdm_Nb_COLA = np.zeros((len(zs_COLA),len(ks_COLA)),'float64')
    T_d_b_Nb_COLA = np.zeros((len(zs_COLA),len(ks_COLA)),'float64')
    T_d_g_Nb_COLA = np.zeros((len(zs_COLA),len(ks_COLA)),'float64')
    T_d_ur_Nb_COLA = np.zeros((len(zs_COLA),len(ks_COLA)),'float64')
    T_d_total_Nb_COLA = np.zeros((len(zs_COLA),len(ks_COLA)),'float64')
    T_d_total_de_COLA = np.zeros((len(zs_COLA),len(ks_COLA)),'float64')

    d_GR_CAMB_at_tau = interp1d(tau,d_GR_CAMB,axis=0)
    d_m_Nb_CAMB_at_tau = interp1d(tau,d_m_Nb_CAMB,axis=0)
    d_b_Nb_CAMB_at_tau = interp1d(tau,d_b_Nb_CAMB,axis=0)
    d_cdm_Nb_CAMB_at_tau = interp1d(tau,d_cdm_Nb_CAMB,axis=0)
    d_g_Nb_CAMB_at_tau = interp1d(tau,d_g_Nb_CAMB,axis=0)
    d_ur_Nb_CAMB_at_tau = interp1d(tau,d_ur_Nb_CAMB,axis=0)
    d_total_Nb_CAMB_at_tau = interp1d(tau,d_total_Nb_CAMB,axis=0)
    d_total_DE_Nb_CAMB_at_tau = interp1d(tau,d_total_DE_Nb_CAMB,axis=0)

    # d_GR has a minus sign due to CLASS/CAMB conventions!

    for index_z in range(len(zs_COLA)):
        T_d_GR_COLA[index_z,:] = -d_GR_CAMB_at_tau(background_tau_at_z(zs_COLA[index_z]))
        T_d_m_Nb_COLA[index_z,:] = d_m_Nb_CAMB_at_tau(background_tau_at_z(zs_COLA[index_z]))
        T_d_cdm_Nb_COLA[index_z,:] = d_cdm_Nb_CAMB_at_tau(background_tau_at_z(zs_COLA[index_z]))
        T_d_b_Nb_COLA[index_z,:] = d_b_Nb_CAMB_at_tau(background_tau_at_z(zs_COLA[index_z]))
        T_d_g_Nb_COLA[index_z,:] = d_g_Nb_CAMB_at_tau(background_tau_at_z(zs_COLA[index_z]))
        T_d_ur_Nb_COLA[index_z,:] = d_ur_Nb_CAMB_at_tau(background_tau_at_z(zs_COLA[index_z]))
        T_d_total_Nb_COLA[index_z,:] = d_total_Nb_CAMB_at_tau(background_tau_at_z(zs_COLA[index_z]))
        T_d_total_de_COLA[index_z,:] = d_total_DE_Nb_CAMB_at_tau(background_tau_at_z(zs_COLA[index_z]))
        T_d_no_nu_Nb_COLA = T_d_m_Nb_COLA
        T_Weyl_de_COLA = np.zeros((len(zs_COLA),len(ks_COLA)),'float64')
        T_v_cdm_COLA = np.zeros((len(zs_COLA),len(ks_COLA)),'float64')
        T_v_b_COLA = np.zeros((len(zs_COLA),len(ks_COLA)),'float64')
        T_v_b_min_v_c_COLA = np.zeros((len(zs_COLA),len(ks_COLA)),'float64')

    # Formatting data
    data = {}
    for index_z in range(len(zs_COLA)):
        #for index_column in range(12):
        data[zs_COLA[index_z]] = np.vstack((ks_COLA, T_d_cdm_Nb_COLA[index_z,:], T_d_b_Nb_COLA[index_z,:],
                                            T_d_g_Nb_COLA[index_z,:], T_d_ur_Nb_COLA[index_z,:],
                                            T_d_GR_COLA[index_z,:], T_d_total_Nb_COLA[index_z,:],
                                            T_d_no_nu_Nb_COLA[index_z,:], T_d_total_de_COLA[index_z,:],
                                            T_Weyl_de_COLA[index_z,:], T_v_cdm_COLA[index_z,:],
                                            T_v_b_COLA[index_z,:], T_v_b_min_v_c_COLA[index_z,:])).T

    # Save transfers in COLA format
    for i in range(len(zs_COLA)):
        np.savetxt(f"{path}/data_transfer_z{zs_COLA[i]:.3f}.dat", data[zs_COLA[i]],
                    fmt='   %.7E   %.7E   %.7E   %.7E   %.7E   %.7E   %.7E   %.7E   %.7E   %.7E   %.7E   %.7E   %.7E ',
                    header='          k/h    CDM            baryon         photon         nu             mass_nu        total          no_nu          total_de       Weyl           v_CDM          v_b            v_b-v_c')
    
    # Write transferinfo file
    cluster_path = f"/gpfs/projects/MirandaGroup/victoria/cola_projects/test/w0waCDM/transfer_functions/{sim_number}/"
    with open(f"{path}/transferinfo.dat", "w") as f:
        f.write(f"{cluster_path} 54\n")
        for i in reversed(range(len(zs_COLA))):
            f.write(f'data_transfer_z{zs_COLA[i]:.3f}.dat  {zs_COLA[i]:.5f}\n')
    print(f"Saved transfer functions and transferinfo in {path}")

# Parsing command line arguments
argp = argparse.ArgumentParser(
    description="Generates transfer functions for COLA simulations",
)
argp.add_argument("--path_to_save", help="Path to save the files", required=True)
argp.add_argument("--input", help="File to read LHS. If not provided, calculates and saves the transfer functions for the EE2 reference cosmology.")
argp.add_argument("--start", help="LHS cosmology index to start generating.", type=int)
argp.add_argument("--end", help="LHS cosmology index to end generating.", type=int)

if __name__ == "__main__":
    print("----- Transfer Function Generator for w0wa -----")
    args = argp.parse_args()
    if not os.path.isdir(args.path_to_save) and not os.path.isfile(args.path_to_save): os.mkdir(args.path_to_save)
    # else:
    #     print(f"ERROR: path {args.path_to_save} is not empty. Aborting.")
    #     exit(1)
    print(f"Will save transfer function data in {args.path_to_save}")
    ref = args.input is None
    if args.start is None: args.start = 0
    else: args.start
    if not ref:
        print(f"Reading LHS file in {args.input}")
        lhs = np.loadtxt(args.input)
        for i, cosmo in enumerate(lhs):
            # if i < args.start or i > args.end: 
            #     print(f"Skipping cosmology {i}...")
            #     print("-----")
            #     continue
            print("-----")
            try:
                h, Omega_b, Omega_m, As, ns, w, wa = cosmo
            except ValueError:
                Omega_m, Omega_b, ns, As, h, w = cosmo
                w0pwa = w
            # wa = w0pwa - w
            print(f'Running cosmology #{i}: Om={Omega_m}, Ob={Omega_b}, ns={ns}, As={As}, h={h}, w0={w}, wa={wa}')
            os.mkdir(f"{args.path_to_save}/{i}")
            start = time.perf_counter()
            generate_transfers(Omega_m, Omega_b, h, As, ns, w, wa, path=f"{args.path_to_save}/{i}", sim_number=i)
            elapsed = time.perf_counter() - start
            print(f"Generation took {elapsed} seconds")
    else:
        print("Calculating Tk for reference cosmo")
        Omega_m_ref = 0.319
        Omega_b_ref = 0.049
        As_ref = 2.1e-9
        ns_ref = 0.96
        h_ref = 0.67
        w_ref = -1
        wa_ref = 0
        start = time.perf_counter()
        generate_transfers(Omega_m_ref, Omega_b_ref, h_ref, As_ref, ns_ref, w_ref, wa_ref, path=args.path_to_save, sim_number="ref")
        elapsed = time.perf_counter() - start
        print(f"Generation took {elapsed:.1f} seconds")
    print("----- Finished! -----")
