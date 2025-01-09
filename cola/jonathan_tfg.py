import argparse
import os
import sys
import numpy as np
from classy import Class
from scipy.interpolate import interp1d

zs_COLA = [0.000, 0.020, 0.041, 0.062, 0.085, 0.109, 0.133, 0.159, 0.186, 0.214, 0.244, 0.275, 0.308, 
           0.342, 0.378, 0.417, 0.457, 0.500, 0.543, 0.588, 0.636, 0.688, 0.742, 0.800, 0.862, 0.929, 
           1.000, 1.087, 1.182, 1.286, 1.400, 1.526, 1.667, 1.824, 2.000, 2.158, 2.333, 2.529, 2.750, 
           3.000, 3.286, 3.615, 4.000, 4.455, 5.000, 5.667, 6.500, 7.571, 9.000, 11.000, 14.000, 19.000, 
           19.500, 20.000]
zs_COLA = np.flip(zs_COLA)

def generate_transfers(Omega_m, Omega_b, h, As, ns, w, wa, base_path, cluster_base_path, filename):
    param_path = base_path + f'cosmo_params/params_{filename}.txt'
    bg_path = base_path + f'backgrounds/bg_{filename}.txt'
    transfer_function_path = base_path + f'transfer_functions/{filename}/'
    transferinfo_path = base_path + f'transferinfo_files/transferinfo{filename}.txt'
    cluster_path = cluster_base_path + f'transfer_functions/{filename}/'

    m_nu = 0.058
    Omega_nu = (m_nu/94.13)/(h*h)
    Omega_c = Omega_m - Omega_b - Omega_nu
    k_per_decade = 25
    z_max_pk = 250

    cosmo_compute = Class()
    cosmo_compute.set({
        'output': 'dTk, vTk, lTk', # lTk is the new approximation scheme transfer functions
        'Omega_b': Omega_b,
        'Omega_cdm': Omega_c,
        'w0_fld': f'{w}',
        'wa_fld': f'{wa}',
        'n_s': ns,
        'A_s': As,
        'h': h,
        'Omega_Lambda': 0,
        'use_ppf': 'yes',
        'radiation_streaming_approximation': 3, # turnoff radiation approximation.
        'ur_fluid_approximation': 2, # turnoff massless nu fluid approximation
        'z_max_pk': z_max_pk,
        'z_pk': '99.0, 9.0, 4.0, 0.',
        'evolver': 0, # 0 runge kutta solver 
        'P_k_max_h/Mpc' : 10.0,
        #'k_min_tau0': 0.2,
        #'perturb_sampling_stepsize': 1e-4, # 0.1e-3, # Needs to be small because PPF eqs are very stiff at large scales, consumes a lot of memory
        'k_per_decade_for_pk': k_per_decade,
        'background_verbose': 0, 
        'perturbations_verbose': 0,
        'gauge': 'Synchronous',
        'l_max_ncdm': 500,  # When creating the transfer functions for our runs uncomment these l_maxs!!!!
        'l_max_g': 3500, # When creating the transfer functions for our runs uncomment these l_maxs!!!!
        'l_max_pol_g': 3500, # When creating the transfer functions for our runs uncomment these l_maxs!!!!
        'l_max_ur': 3500, # When creating the transfer functions for our runs uncomment these l_maxs!!!!
        'N_ur': 0, #number of ultra relativistic species masless nus
        'N_ncdm': 1, #number of non-cold dark matter species massive nus
        'm_ncdm': m_nu/3, #mass for ncdm species
        'deg_ncdm': 3.0, #number of degrees of freedom of massive nus
        'ncdm_fluid_approximation': 3
    })
    cosmo_compute.compute()
    
    background = cosmo_compute.get_background()

    tau_num = 1000
    tau = np.linspace(1000, background['conf. time [Mpc]'][-1], tau_num)

    background_tau = background['conf. time [Mpc]'] # read conformal times in background table
    background_z = background['z'] # read redshift
    background_a = 1./(1.+background_z) # read redshift
    background_H = background['H [1/Mpc]']

    background_a_at_tau = interp1d(background_tau, background_a, fill_value="extrapolate")
    background_z_at_tau = interp1d(background_tau, background_z, fill_value="extrapolate")
    background_tau_at_z = interp1d(background_z, background_tau, fill_value="extrapolate")
    background_rho_GR_at_tau = interp1d(background_tau, background['(.)rho_ncdm[0]'] + background['(.)rho_g'] + background['(.)rho_fld'],fill_value="extrapolate")

    # Saving cosmology params
    Omega_massivenu_exact = background['(.)rho_ncdm[0]'][-1]/background['(.)rho_crit'][-1]
    Omega_c_exact = background['(.)rho_cdm'][-1]/background['(.)rho_crit'][-1]
    Omega_b_exact = background['(.)rho_b'][-1]/background['(.)rho_crit'][-1]
    Omega_m_exact = Omega_b_exact + Omega_c_exact + Omega_massivenu_exact

    with open(param_path, 'w') as f:
        f.write("# ORDER: Omega_nu, Omega_c, Omega_b, Omega_m, As, ns, w0, h, wa\n")
        f.write(f'{Omega_massivenu_exact}\n')
        f.write(f'{Omega_c_exact}\n')
        f.write(f'{Omega_b_exact}\n')
        f.write(f'{Omega_m_exact}\n')
        f.write(f'{As}\n')
        f.write(f'{ns}\n')
        f.write(f'{w}\n')
        f.write(f'{h}\n')
        f.write(f'{wa}\n')

    max_z_needed = background_z_at_tau(tau[0])
    if max_z_needed > z_max_pk:
        print(f'ERROR: you must increase the value of z_max_pk to at least {max_z_needed}')
        exit(1)

    # get transfer functions at each time and build arrays Theta0(tau,k) and phi(tau,k)
    for i in range(tau_num):

        this_z = background_z_at_tau(tau[i])
        if this_z < 0:
            print(f"WARNING: interpolation error led to negative redshift {this_z} at tau = {tau[i]}. Correcting to z = 0.")
            this_z = 0

        one_time = cosmo_compute.get_transfer(this_z) # transfer functions at each time tau
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
            d_rhoGR_Nb = np.zeros((tau_num,k_num))
            HT_Nb_pp_num = np.zeros((tau_num,k_num))
            HT_Nb_pp = np.zeros((tau_num,k_num))
            sources_Nb = np.zeros((tau_num,k_num))


        d_m_Nb_CAMB[i,:] = one_time['d_m_Nb_CAMB'][:]
        d_cdm_Nb_CAMB[i,:] = one_time['d_c_Nb_CAMB'][:]
        d_b_Nb_CAMB[i,:] = one_time['d_b_Nb_CAMB'][:]
        d_g_Nb_CAMB[i,:] = one_time['d_g_Nb_CAMB'][:]
        d_ur_Nb_CAMB[i,:] = one_time['d_ur_Nb_CAMB'][:]
        d_total_Nb_CAMB[i,:] = one_time['delta_total_Nb_CAMB'][:]
        d_total_DE_Nb_CAMB[i,:] = one_time['delta_total_DE_Nb_CAMB'][:]
        d_rhoGR_Nb[i,:] = one_time['d_rhoGR_Nb'][:]
        HT_Nb_pp_num[i,:] = one_time['H_T_Nb_prime_prime_num'][:]
        HT_Nb_pp[i,:] = one_time['H_T_Nb_prime_prime'][:]
        sources_Nb[i,:] = 1.5 * background_a_at_tau(tau[i])**2. * one_time['sources_Nb'][:]    
        d_GR_CAMB[i,:] = (-(-one_time['k2gamma_Nb_num'][:]+sources_Nb[i,:])/background_rho_GR_at_tau(tau[i])/k/k/cosmo_compute.h()/cosmo_compute.h())*cosmo_compute.h()

    ks_COLA = one_time['k (h/Mpc)']

    T_d_GR_COLA = np.zeros((len(zs_COLA),len(ks_COLA)),'float64') 
    T_d_m_Nb_COLA = np.zeros((len(zs_COLA),len(ks_COLA)),'float64')
    T_d_cdm_Nb_COLA = np.zeros((len(zs_COLA),len(ks_COLA)),'float64')
    T_d_b_Nb_COLA = np.zeros((len(zs_COLA),len(ks_COLA)),'float64')
    T_d_g_Nb_COLA = np.zeros((len(zs_COLA),len(ks_COLA)),'float64')
    T_d_ur_Nb_COLA = np.zeros((len(zs_COLA),len(ks_COLA)),'float64')
    T_d_total_Nb_COLA = np.zeros((len(zs_COLA),len(ks_COLA)),'float64')
    T_d_total_de_COLA = np.zeros((len(zs_COLA),len(ks_COLA)),'float64')

    #T_Geff = np.zeros((len(zs_COLA),len(ks_COLA)),'float64') 


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
        #T_Geff[index_z,:] = np.ones(len(ks_COLA))
        T_d_no_nu_Nb_COLA = T_d_m_Nb_COLA
        T_Weyl_de_COLA = np.zeros((len(zs_COLA),len(ks_COLA)),'float64')
        T_v_cdm_COLA = np.zeros((len(zs_COLA),len(ks_COLA)),'float64')
        T_v_b_COLA = np.zeros((len(zs_COLA),len(ks_COLA)),'float64')
        T_v_b_min_v_c_COLA = np.zeros((len(zs_COLA),len(ks_COLA)),'float64')


    #T_d_no_no is just T_d_m_Nb
    data = {}
    for index_z in range(len(zs_COLA)):
        #for index_column in range(12):
        data[zs_COLA[index_z]] = np.vstack((ks_COLA, T_d_cdm_Nb_COLA[index_z,:], T_d_b_Nb_COLA[index_z,:],
                                            T_d_g_Nb_COLA[index_z,:], T_d_ur_Nb_COLA[index_z,:],
                                            T_d_GR_COLA[index_z,:], T_d_total_Nb_COLA[index_z,:],
                                            T_d_no_nu_Nb_COLA[index_z,:], T_d_total_de_COLA[index_z,:],
                                            T_Weyl_de_COLA[index_z,:], T_v_cdm_COLA[index_z,:],
                                            T_v_b_COLA[index_z,:], T_v_b_min_v_c_COLA[index_z,:])).T

    # Routine to save a bunch of files in the COLA format
    try:
        os.mkdir(transfer_function_path)
    except FileExistsError:
        pass 

    for i in range(len(zs_COLA)):
        np.savetxt(f"{transfer_function_path}/data_transfer_z{zs_COLA[i]:.3f}.dat", data[zs_COLA[i]],
                    fmt='   %.7E   %.7E   %.7E   %.7E   %.7E   %.7E   %.7E   %.7E   %.7E   %.7E   %.7E   %.7E   %.7E ',
                    header='          k/h    CDM            baryon         photon         nu             mass_nu        total          no_nu          total_de       Weyl           v_CDM          v_b            v_b-v_c')
    
    # Routine to print a lot if files names for that infofile we pass to COLA
    with open(transferinfo_path, "w") as f:
        f.write(cluster_path + f' {len(zs_COLA)}\n')
        for i in reversed(range(len(zs_COLA))):
            fname = f"data_transfer_z{zs_COLA[i]:.3f}.dat"
            f.write(fname + '  ' + str(np.round(zs_COLA[i],5))+'\n')
    print(f"Saved transfer functions in {transfer_function_path}")

base_path = './w0wa/'
cluster_base_path = '/gpfs/projects/MirandaGroup/jonathan/cola_projects/lcdm_proj_tests/w0wa/'
lhs_path = base_path + 'lhs.txt'

nmin = int(sys.argv[1])
nmax = int(sys.argv[2])
ref=False
print(f"Will save transfer function data in {base_path}")
if not ref:
    print(f"Reading LHS file in {lhs_path}")
    lhs = np.loadtxt(lhs_path)
    for i in range(nmin, nmax):
        cosmo=lhs[i]
        Omega_m, Omega_b, ns, As, h, w, w0pwa = cosmo
        wa = w0pwa - w

        print(f'Running cosmology #{i}: Om={Omega_m}, Ob={Omega_b}, ns={ns}, As={As}, h={h}, w0={w}, wa={wa}')
        generate_transfers(Omega_m, Omega_b, h, As, ns, w, wa, base_path, cluster_base_path, str(i))
else:
    Omega_m_ref = 0.319
    Omega_b_ref = 0.049
    As_ref = 2.1e-9
    ns_ref = 0.96
    h_ref = 0.67
    w_ref = -1
    wa_ref = 0
    print(f'Running ref cosmology: Om={Omega_m_ref}, Ob={Omega_b_ref}, ns={ns_ref}, As={As_ref}, h={h_ref}, w0={w_ref}, wa={wa_ref}')
    generate_transfers(Omega_m_ref, Omega_b_ref, h_ref, As_ref, ns_ref, w_ref, wa_ref, base_path, cluster_base_path, 'ref')

