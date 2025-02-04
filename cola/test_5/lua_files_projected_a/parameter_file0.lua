all_parameters_must_be_in_file = true

particle_Npart_1D = 1024
force_nmesh = 2048
ic_random_seed = 1234
output_redshifts = {3, 2, 1, 0.5, 0.0}
timestep_nsteps = {12, 5, 8, 9, 17}
output_particles = false
fof = false
fof_nmin_per_halo = 20
pofk = true
pofk_nmesh = 1024
ic_fix_amplitude = true
ic_reverse_phases = false
ic_type_of_input = "transferinfofile"
ic_input_filename = "/gpfs/projects/MirandaGroup/victoria/cola_projects/test_5/transfers_projected/0/transferinfo.dat"

output_folder = "/gpfs/projects/MirandaGroup/victoria/cola_projects/test_5/output/projected_a/0"
simulation_name = "run_0"
simulation_boxsize = 1024

simulation_use_cola = true
simulation_use_scaledependent_cola = true

simulation_enforce_LPT_trajectories = false

simulation_cola_LPT_order = 2

cosmology_model = "LCDM"
cosmology_OmegaCDM = 0.28165999213174986
cosmology_Omegab = 0.03965160937629297
cosmology_OmegaMNu = 0.0013847270112202958
cosmology_OmegaLambda = 1 - cosmology_OmegaCDM - cosmology_Omegab - cosmology_OmegaMNu
cosmology_Neffective = 3.046
cosmology_TCMB_kelvin = 2.7255
cosmology_h = 0.667064586976158
cosmology_As = 1.7871747151195836e-09
cosmology_ns = 0.9987641477110053
cosmology_kpivot_mpc = 0.05
cosmology_OmegaK = 0.0
if cosmology_model == "w0waCDM" then 
  cosmology_w0 = -1
  cosmology_wa = 0
end

if cosmology_model == "DGP" then 
  cosmology_dgp_OmegaRC = 0.11642
end

gravity_model = "GR"

if gravity_model == "f(R)" then 
  gravity_model_fofr_fofr0 = 1e-5
  gravity_model_fofr_nfofr = 1.0
  gravity_model_screening = true
  gravity_model_screening_enforce_largescale_linear = true
  gravity_model_screening_linear_scale_hmpc = 0.1
end

if gravity_model == "DGP" then 
  gravity_model_dgp_rcH0overc = 1.0
  gravity_model_screening = true
  gravity_model_dgp_smoothing_filter = "tophat"
  gravity_model_dgp_smoothing_scale_over_boxsize = 0.0
  gravity_model_screening_enforce_largescale_linear = true
  gravity_model_screening_linear_scale_hmpc = 0.1
end

particle_allocation_factor = 1.25

output_fileformat = "GADGET"

timestep_method = "Quinn"
timestep_cola_nLPT = -2.5
timestep_algorithm = "KDK"
timestep_scalefactor_spacing = "linear"
if timestep_scalefactor_spacing == "powerlaw" then
  timestep_spacing_power = 1.0
end

ic_random_generator = "GSL"
ic_random_field_type = "gaussian"
ic_nmesh = particle_Npart_1D
ic_use_gravity_model_GR = true
ic_LPT_order = 2
ic_input_redshift = 19.0
ic_initial_redshift = 19.0
ic_sigma8_normalization = false
ic_sigma8_redshift = 0.0
ic_sigma8 = 0.83

ic_type_of_input_fileformat = "CAMB"
if ic_random_field_type == "nongaussian" then
  ic_fnl_type = "local"
  ic_fnl = 100.0
  ic_fnl_redshift = ic_initial_redshift
end

if ic_random_field_type == "reconstruct_from_particles" then
  ic_reconstruct_gadgetfilepath = "output/gadget"
  ic_reconstruct_assigment_method = "CIC"
  ic_reconstruct_smoothing_filter = "sharpk"
  ic_reconstruct_dimless_smoothing_scale = 1.0 / (128 * math.pi)
  ic_reconstruct_interlacing = false
end

if ic_random_field_type == "read_particles" then
  ic_reconstruct_gadgetfilepath = "path/gadget"
end

force_density_assignment_method = "TSC"
-- force_kernel = "continuous_greens_function"
force_use_finite_difference_force = true
force_finite_difference_stencil_order = 4
force_greens_function_kernel = "discrete_2pt"
force_gradient_kernel = "discrete_4pt"
force_linear_massive_neutrinos = true

lightcone = false

fof_linking_length = 0.2 / particle_Npart_1D
fof_nmesh_max = 0

pofk_interlacing = true
pofk_subtract_shotnoise = false
pofk_density_assignment_method = "PCS"

pofk_multipole = false
pofk_multipole_nmesh = 128
pofk_multipole_interlacing = true
pofk_multipole_subtract_shotnoise = false
pofk_multipole_ellmax = 4
pofk_multipole_density_assignment_method = "PCS"

bispectrum = false
bispectrum_nmesh = 128
bispectrum_nbins = 10
bispectrum_interlacing = true
bispectrum_subtract_shotnoise = false
bispectrum_density_assignment_method = "PCS"
