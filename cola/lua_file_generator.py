import os
import argparse
import numpy as np

def write_lua_file(cosmo, ident, lua_out_path, transferinfo_path, output_path, pair_fix, phase, projected):
    """
        Description: function that writes a .lua file that serves as COLA-FML input.
        Arguments:
            cosmo: a list with cosmological parameters. Must have shape (N, 5), (N, 6) or (N, 7)
            ident: a tag for the simulation
            lua_out_path: where you want the lua file to be saved
            transferinfo_path: the path for the transferinfo file in the cluster
            output_path: the path where the simulations are going to be saved in the cluster
            pair_fix: whether to fix amplitudes of the initial conditions
            phase: if initial conditions have fixed amplitudes, choose which phase (normal "a" or reversed "b")
            projected: whether to write the lua file for the LCDM projection of the cosmology `cosmo`
    """
    dims = len(cosmo)
    if dims == 5:
        cola_model = "LCDM"
        h, Omega_b, Omega_m, As, ns = cosmo
        w = -1
        wa = 0
    elif dims == 6:
        cola_model = "w0waCDM"
        h, Omega_b, Omega_m, As, ns, w = cosmo
        wa = 0
    elif dims == 7:
        cola_model = "w0waCDM"
        h, Omega_b, Omega_m, As, ns, w, wa = cosmo
    
    if projected:
        # Override with LCDM parameters
        cola_model = "LCDM"
        w = -1
        wa = 0

    Omega_nu = (m_nu/94.13)/(h*h)
    Omega_c = Omega_m - Omega_b - Omega_nu
    
    if not os.path.isdir(f"{lua_out_path}"): os.mkdir(f"{lua_out_path}")
    file = open(f"{lua_out_path}/parameter_file{ident}.lua", 'w')
    
    file.write('all_parameters_must_be_in_file = true\n\n')
    file.write('particle_Npart_1D = 1024\n')
    file.write(f'force_nmesh = {force_nmesh}\n')
    file.write('ic_random_seed = 1234\n')
    file.write('output_redshifts = {3, 2, 1, 0.5, 0.0}\n')
    file.write('timestep_nsteps = {12, 5, 8, 9, 17}\n')
    file.write('output_particles = false\n')
    file.write('fof = false\n')
    file.write('fof_nmin_per_halo = 20\n')
    file.write('pofk = true\n')
    file.write('pofk_nmesh = 1024\n')
    file.write(f'ic_fix_amplitude = {"true" if pair_fix else "false"}\n')
    file.write(f'ic_reverse_phases = {"true" if phase == "b" else "false"}\n')
    file.write('ic_type_of_input = "transferinfofile"\n')
    file.write('ic_input_filename = "' + transferinfo_path + '"\n\n')
    file.write('output_folder = "' + output_path + '"\n')
    file.write(f'simulation_name = "run_{ident}"\n')
    file.write(f'simulation_boxsize = {box_size}\n\n')
    file.write('simulation_use_cola = true\n')
    file.write('simulation_use_scaledependent_cola = true\n\n')
    # new settings
    file.write('simulation_enforce_LPT_trajectories = false\n\n')
    file.write('simulation_cola_LPT_order = 2\n\n')
    ##
    file.write(f'cosmology_model = "{cola_model}"\n')
    file.write(f'cosmology_OmegaCDM = {Omega_c}\n')
    file.write(f'cosmology_Omegab = {Omega_b}\n')
    file.write(f'cosmology_OmegaMNu = {Omega_nu}\n')
    file.write('cosmology_OmegaLambda = 1 - cosmology_OmegaCDM - cosmology_Omegab - cosmology_OmegaMNu\n')
    file.write('cosmology_Neffective = 3.046\n')
    file.write('cosmology_TCMB_kelvin = 2.7255\n')
    file.write(f'cosmology_h = {h}\n')
    file.write(f'cosmology_As = {As}\n')
    file.write(f'cosmology_ns = {ns}\n')
    file.write('cosmology_kpivot_mpc = 0.05\n')
    # new settings
    file.write('cosmology_OmegaK = 0.0\n')
    ##
    file.write('if cosmology_model == "w0waCDM" then \n')
    file.write(f'  cosmology_w0 = {w}\n')
    file.write(f'  cosmology_wa = {wa}\n')
    file.write('end\n\n')
    file.write('if cosmology_model == "DGP" then \n')
    file.write('  cosmology_dgp_OmegaRC = 0.11642\n')
    file.write('end\n\n')
    file.write('gravity_model = "GR"\n\n')
    file.write('if gravity_model == "f(R)" then \n')
    file.write('  gravity_model_fofr_fofr0 = 1e-5\n')
    file.write('  gravity_model_fofr_nfofr = 1.0\n')
    file.write('  gravity_model_screening = true\n')
    file.write('  gravity_model_screening_enforce_largescale_linear = true\n')
    file.write('  gravity_model_screening_linear_scale_hmpc = 0.1\n')
    file.write('end\n\n')
    file.write('if gravity_model == "DGP" then \n')
    file.write('  gravity_model_dgp_rcH0overc = 1.0\n')
    file.write('  gravity_model_screening = true\n')
    file.write('  gravity_model_dgp_smoothing_filter = "tophat"\n')
    file.write('  gravity_model_dgp_smoothing_scale_over_boxsize = 0.0\n')
    file.write('  gravity_model_screening_enforce_largescale_linear = true\n')
    file.write('  gravity_model_screening_linear_scale_hmpc = 0.1\n')
    file.write('end\n\n')
    file.write('particle_allocation_factor = 1.25\n\n')
    file.write('output_fileformat = "GADGET"\n\n')
    file.write('timestep_method = "Quinn"\n')
    file.write('timestep_cola_nLPT = -2.5\n')
    file.write('timestep_algorithm = "KDK"\n')
    file.write('timestep_scalefactor_spacing = "linear"\n')
    file.write('if timestep_scalefactor_spacing == "powerlaw" then\n')
    file.write('  timestep_spacing_power = 1.0\n')
    file.write('end\n\n')
    file.write('ic_random_generator = "GSL"\n')
    file.write('ic_random_field_type = "gaussian"\n')
    file.write('ic_nmesh = particle_Npart_1D\n')
    file.write('ic_use_gravity_model_GR = true\n')
    file.write('ic_LPT_order = 2\n')
    file.write(f'ic_input_redshift = {z_ini}\n')
    file.write(f'ic_initial_redshift = {z_ini}\n')
    file.write('ic_sigma8_normalization = false\n')
    file.write('ic_sigma8_redshift = 0.0\n')
    file.write('ic_sigma8 = 0.83\n\n')
    # new settings
    file.write('ic_type_of_input_fileformat = "CAMB"\n')
    ##
    file.write('if ic_random_field_type == "nongaussian" then\n')
    file.write('  ic_fnl_type = "local"\n')
    file.write('  ic_fnl = 100.0\n')
    file.write('  ic_fnl_redshift = ic_initial_redshift\n')
    file.write('end\n\n')
    file.write('if ic_random_field_type == "reconstruct_from_particles" then\n')
    file.write('  ic_reconstruct_gadgetfilepath = "output/gadget"\n')
    file.write('  ic_reconstruct_assigment_method = "CIC"\n')
    file.write('  ic_reconstruct_smoothing_filter = "sharpk"\n')
    file.write('  ic_reconstruct_dimless_smoothing_scale = 1.0 / (128 * math.pi)\n')
    file.write('  ic_reconstruct_interlacing = false\n')
    file.write('end\n\n')
    file.write('if ic_random_field_type == "read_particles" then\n')
    file.write('  ic_reconstruct_gadgetfilepath = "path/gadget"\n')
    file.write('end\n\n')
    # changed settings
    file.write('force_density_assignment_method = "TSC"\n')
    file.write('-- force_kernel = "continuous_greens_function"\n')
    # new settings
    file.write('force_use_finite_difference_force = true\n')
    file.write('force_finite_difference_stencil_order = 4\n')
    file.write('force_greens_function_kernel = "discrete_2pt"\n')
    file.write('force_gradient_kernel = "discrete_4pt"\n')
    ##
    file.write('force_linear_massive_neutrinos = true\n\n')
    # new settings
    file.write('lightcone = false\n\n')
    ##
    file.write('fof_linking_length = 0.2 / particle_Npart_1D\n')
    file.write('fof_nmesh_max = 0\n\n')
    file.write('pofk_interlacing = true\n')
    file.write('pofk_subtract_shotnoise = false\n')
    file.write('pofk_density_assignment_method = "PCS"\n\n')
    file.write('pofk_multipole = false\n')
    file.write('pofk_multipole_nmesh = 128\n')
    file.write('pofk_multipole_interlacing = true\n')
    file.write('pofk_multipole_subtract_shotnoise = false\n')
    file.write('pofk_multipole_ellmax = 4\n')
    file.write('pofk_multipole_density_assignment_method = "PCS"\n\n')
    file.write('bispectrum = false\n')
    file.write('bispectrum_nmesh = 128\n')
    file.write('bispectrum_nbins = 10\n')
    file.write('bispectrum_interlacing = true\n')
    file.write('bispectrum_subtract_shotnoise = false\n')
    file.write('bispectrum_density_assignment_method = "PCS"\n')
    file.close()

# --- Main program ---

argp = argparse.ArgumentParser(
    description="Generates lua input files for COLA simulations",
)
argp.add_argument("--path_to_save", help="Path to save the files.", required=True)
argp.add_argument("--precision", help="Simulation precision, controls the size of the box, number of particles and size of mesh grid. Supported values are 'default' and 'high'.", required=True, choices=["default", "high"])
argp.add_argument("--input", help="File to read LHS. If not provided, generates lua files for the EE2 reference cosmology.")
argp.add_argument("--technique", help="Whether to fix amplitudes and reverse the phase of initial conditions in this batch. This option generates two lua files for the same cosmology, one for each phase.", required=True, choices=["pair-fix", "project"])

z_ini = 19.0
m_nu = 0.058

if __name__ == "__main__":
    args = argp.parse_args()

    if args.precision == 'default':
        force_nmesh = 2048
        box_size = 1024
        prec_suffix = '1'
    elif args.precision == 'high':
        force_nmesh = 3072
        box_size = 512
        prec_suffix = '2'

    ref = args.input is None
    if not ref:
        try:
            lhs = np.loadtxt(args.input)
        except Exception as err:
            print(f"ERROR: could not open LHS file in {args.input} due to the following error: {err}")
            exit(1)

        if len(lhs.shape) != 2 or lhs.shape[1] not in [5, 6, 7]:
            print(f"ERROR: invalid LHS format. Should be (N, 5) for LCDM, (N, 6) for wCDM or (N, 7) for w0wa, got {lhs.shape}")
            exit(1)    
    else:
        lhs = np.array([[0.67, 0.049, 0.319, 2.1e-9, 0.96]])
    
    num_points, dims = lhs.shape
    
    if args.technique == "pair-fix":
        for i, cosmo in enumerate(lhs):
            for phase in ["a", "b"]:
                # Where on the current computer to store the lua files this script writes
                lua_out_path = f'{args.path_to_save}/lua_files_{phase}/' if not ref else f'{args.path_to_save}/lua_files_{phase}/'
                # Where the transferinfo files will be read from on the cluster
                transferinfo_path = f'/gpfs/projects/MirandaGroup/victoria/cola_projects/test_5/transfers/{i}/transferinfo.dat'
                # Where the output pk will be stored on the cluster
                output_path = f'/gpfs/projects/MirandaGroup/victoria/cola_projects/test_5/output/{phase}/{i}'
                write_lua_file(cosmo, i, lua_out_path, transferinfo_path, output_path, pair_fix=True, phase=phase, projected=False)
    elif args.technique == "project":
        for i, cosmo in enumerate(lhs):
            for projected in [False, True]:
                # Where on the current computer to store the lua files this script writes
                lua_out_path = f'{args.path_to_save}/lua_files_no_pairfix{"_projected" if projected else ""}/' if not ref else f'{args.path_to_save}/lua_files_no_pairfix{"_projected" if projected else ""}/'
                # Where the transferinfo files will be read from on the cluster
                transferinfo_path = f'/gpfs/projects/MirandaGroup/victoria/cola_projects/test_5/transfers{"_projected" if projected else ""}/{i}/transferinfo.dat'
                # Where the output pk will be stored on the cluster
                output_path = f'/gpfs/projects/MirandaGroup/victoria/cola_projects/test_5/output/no_pairfix{"_projected" if projected else ""}/{i}'
                write_lua_file(cosmo, i, lua_out_path, transferinfo_path, output_path, pair_fix=False, phase="a", projected=projected)