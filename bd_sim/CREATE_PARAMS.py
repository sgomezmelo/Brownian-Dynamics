import numpy as np
from numpy import linalg as LA
def create_params(TEMP=293.15, VISC=1e-3,
                  REP=100, TREP=20, ATT=300, ANG=0.3,
                  BOXLENGTH=1000e-9, RCATT=2.5,
                  COMPC=2.5e-9,
                  PARTNUM=60, SIMSTEPS=100000,
                  SAVEINT=500, SAVE_ALL_DATA=False,
                  GRAPHICAL=False, TAIL=False, ONLY_STATISTICS=False, ASSEMBL_DETAILS=False, ONLY_TAIL_ANGLE=False,
                  MAIN_PATH="DataAnalysis_Cluster/", SUB_PATH="Results/" , N_trials = 5):




    '''
        Contains all relevant simulation parameters.
        There are two kind of quantities:
        The "real" quantities i.e. with physical units (SI).
        Non dimensional quantites (re-scaled using the defined scales _SCALE) are always written with a _D (dimensionless)

    '''
    # constants scales
    TEMPERATURE = TEMP  # [T] = K
    h = 6.63e-34  # Plancks Constant in J.s for thermal de Broglie
    BOLTZMANN_CONSTANT = 1.38e-23  # [k_B] = J/K
    THERMAL_ENERGY = TEMPERATURE * BOLTZMANN_CONSTANT  # kT
    BETA = 1 / THERMAL_ENERGY
    VISCOSITY = VISC  # [eta] = Ns/m^2
    WCA_REPULSIVE_ENERGY_PARAMETER = REP * BOLTZMANN_CONSTANT * TEMPERATURE
    ATTRACTIVE_ENERGY_STRENGTH = ATT * BOLTZMANN_CONSTANT * TEMPERATURE
    TAIL_REPULSIVE_ENERGY_PARAMETER = TREP * BOLTZMANN_CONSTANT * TEMPERATURE
    MASS = 8.573e-23  # Mass of SAS-6 in kg
    V_s = 175e-3 #Surface potential difference in Volts
    q = 4.2e-18  #Approx charge of the SAS-6 in C
    c_res = 0.01 #Concentration of the reservoir in mol/m^3
    Na = 6.022e+23 #Avogadros number
    rho_r = c_res*Na  # Particle density/concentration
    N_m = 2000  # Number of time steps in which a particle creation event is attempted
    
    '''TUNE'''
    # cutoffs
    REPULSIVE_CUTOFF = 8e-9
    REPULSIVE_CUTOFF_TAIL_COM = 5.9e-9
    REPULSIVE_CUTOFF_TAILS = 3.5e-9
    INIT_CUTOFF_TAILS = 12.82e-9
    CHARACTERISTIC_LENGTH_SCALE_POTENTIAL = REPULSIVE_CUTOFF / (2 ** (1 / 6))
    CHARACTERISTIC_LENGTH_SCALE_POTENTIAL_TAILS = REPULSIVE_CUTOFF_TAILS / (2 ** (1 / 6))
    CHARACTERISTIC_LENGTH_SCALE_POTENTIAL_TAIL_COM = REPULSIVE_CUTOFF_TAIL_COM / (2 ** (1 / 6))

    

    print("CHARACTERISTIC_LENGTH_SCALE_POTENTIAL", CHARACTERISTIC_LENGTH_SCALE_POTENTIAL)
    ATTRACTIVE_CUTOFF = RCATT * CHARACTERISTIC_LENGTH_SCALE_POTENTIAL - REPULSIVE_CUTOFF # 0.2e-9
    print("ATTRACTIVE_CUTOFF", ATTRACTIVE_CUTOFF)
    R_C = RCATT * CHARACTERISTIC_LENGTH_SCALE_POTENTIAL
    ANGULAR_CUTOFF = ANG * np.pi
    COMPLEX_CUTOFF = COMPC
    print("COMPLEX_CUTOFF", COMPLEX_CUTOFF)
    # CUTOFF_ANGLE = 0.2 * np.pi

    # geometry, patch vectors and patch-alignment vectors with physical units
    PATCH_VECTOR_1 = np.array([-3.8375e-9, -1.4008e-9]) #np.array([-3.8375e-9, -2.7894e-9]) non spherical
    PATCH_VECTOR_2 = np.array([3.8375e-9, -1.4008e-9])  #np.array([3.8375e-9, -2.7894e-9])

    PATCH_ALIGNMENT_VECTOR_1 = np.array([-3.8375e-9, -1.4008e-9]) #np.array([-1.8591e-9, -0.6767e-9]) #np.array([-1.8591e-9, -0.6767e-9])
    PATCH_ALIGNMENT_VECTOR_2 = np.array([3.8375e-9, -1.4008e-9])  #np.array([1.8591e-9, -0.6767e-9])  #np.array([1.8591e-9, -0.6767e-9])

    PATCH_ALIGNMENT_VECTOR_1_D = PATCH_ALIGNMENT_VECTOR_1 / (LA.norm(PATCH_ALIGNMENT_VECTOR_1))
    PATCH_ALIGNMENT_VECTOR_2_D = PATCH_ALIGNMENT_VECTOR_2 / (LA.norm(PATCH_ALIGNMENT_VECTOR_2))
    angle_patch1_to_patch2 = np.arccos(
        np.dot(PATCH_ALIGNMENT_VECTOR_1_D, PATCH_ALIGNMENT_VECTOR_2_D))  # = 2.4434 = 140°
    print(angle_patch1_to_patch2)
    ''' TUNE'''
    #simulation parameters in physical units
    SIMULATION_AREA_LENGTH = BOXLENGTH
    TIME_STEP = 2.5e-11  # measured in seconds
    DIFFUSION_COEFFICIENT_X = 6.621e-11 #6.632e-11  # [D_x] = m^2/s
    DIFFUSION_COEFFICIENT_Y = 6.972e-11 #6.945e-11  # [D_y] = m^2/s
    DIFFUSION_COEFFICIENT_ROT = 2.094e6 #2.084e6 # [D_r] = 1/s

    # NON-DIMENSIONAL QUANTITIES USED IN THE SIMULATION
    # scales
    LENGTH_SCALE = CHARACTERISTIC_LENGTH_SCALE_POTENTIAL # in m
    TIME_SCALE = (CHARACTERISTIC_LENGTH_SCALE_POTENTIAL**2)/(48*DIFFUSION_COEFFICIENT_X)
    # TIME_SCALE = (8 * np.pi * VISCOSITY * (LENGTH_SCALE ** 3)) / (BOLTZMANN_CONSTANT * TEMPERATURE)
    ENERGY_SCALE = BOLTZMANN_CONSTANT * TEMPERATURE
    FORCE_SCALE = ENERGY_SCALE / LENGTH_SCALE
    TORQUE_SCALE = LENGTH_SCALE * FORCE_SCALE

    # Dimensionless quantities ()_D
    BETA_D = BETA * ENERGY_SCALE
    h_D = h/(ENERGY_SCALE*TIME_SCALE)
    MASS_D = MASS*LENGTH_SCALE**2/(ENERGY_SCALE**2*TIME_SCALE**2)
    V_SURFACE_D = q*V_s/ENERGY_SCALE
    RHO_RES_D  =  rho_r*LENGTH_SCALE**3

    DIFFUSION_COEFFICIENT_X_D = DIFFUSION_COEFFICIENT_X * (TIME_SCALE / (LENGTH_SCALE ** 2))
    DIFFUSION_COEFFICIENT_Y_D = DIFFUSION_COEFFICIENT_Y * (TIME_SCALE / (LENGTH_SCALE ** 2))
    DIFFUSION_COEFFICIENT_ROT_D = DIFFUSION_COEFFICIENT_ROT * TIME_SCALE

    TIME_STEP_D = TIME_STEP / TIME_SCALE


    STANDARD_DEVIATION_X_D = np.sqrt(2 * DIFFUSION_COEFFICIENT_X_D * TIME_STEP_D)
    STANDARD_DEVIATION_Y_D = np.sqrt(2 * DIFFUSION_COEFFICIENT_Y_D * TIME_STEP_D)
    STANDARD_DEVIATION_ROT_D = np.sqrt(2 * DIFFUSION_COEFFICIENT_ROT_D * TIME_STEP_D)

    # sphere model in PFF
    sp1_x, sp1_y = np.array([-1.9784e-9, -0.7241e-9]) / LENGTH_SCALE
    sp2_x, sp2_y = np.array([1.9784e-9, -0.7241e-9]) / LENGTH_SCALE
    sp3_x, sp3_y = np.array([0., 1.7427e-9]) / LENGTH_SCALE
    sp4_x, sp4_y = np.array([0., 4.1077e-9]) / LENGTH_SCALE
    sp5_x, sp5_y = np.array([0., 6.4727e-9]) / LENGTH_SCALE
    ssp_x, ssp_y = np.array([0., 5.87e-9]) / LENGTH_SCALE
    RADIUS_BIG_SPHERE_D = 1.979e-9 / LENGTH_SCALE
    RADIUS_SMALL_SPHERE_D = 1.182e-9 / LENGTH_SCALE

    # dimensionless geometry of patches
    PATCH_VECTOR_1_D = PATCH_VECTOR_1 / LENGTH_SCALE
    PATCH_VECTOR_2_D = PATCH_VECTOR_2 / LENGTH_SCALE
    # print(LA.norm(PATCH_VECTOR_1_D))
    # x and y coordinates of the two patches
    PATCH1_X_D = PATCH_VECTOR_1_D[0]
    PATCH1_Y_D = PATCH_VECTOR_1_D[1]
    PATCH2_X_D = PATCH_VECTOR_2_D[0]
    PATCH2_Y_D = PATCH_VECTOR_2_D[1]

    # x and y coordinates of the two patch-alignment vectors
    PATCH1_ALIGNMENT_X_D = PATCH_ALIGNMENT_VECTOR_1_D[0]
    PATCH1_ALIGNMENT_Y_D = PATCH_ALIGNMENT_VECTOR_1_D[1]
    PATCH2_ALIGNMENT_X_D = PATCH_ALIGNMENT_VECTOR_2_D[0]
    PATCH2_ALIGNMENT_Y_D = PATCH_ALIGNMENT_VECTOR_2_D[1]

    TAIL_X_D = 0.0
    TAIL_Y_D = 1.0

    # non-dimensional parameters for the potentials
    CHARACTERISTIC_LENGTH_SCALE_POTENTIAL_D = CHARACTERISTIC_LENGTH_SCALE_POTENTIAL / LENGTH_SCALE
    CHARACTERISTIC_LENGTH_SCALE_POTENTIAL_TAILS_D = CHARACTERISTIC_LENGTH_SCALE_POTENTIAL_TAILS / LENGTH_SCALE
    CHARACTERISTIC_LENGTH_SCALE_POTENTIAL_TAIL_COM_D = CHARACTERISTIC_LENGTH_SCALE_POTENTIAL_TAIL_COM / LENGTH_SCALE
    REPULSIVE_CUTOFF_D = REPULSIVE_CUTOFF / LENGTH_SCALE
    REPULSIVE_CUTOFF_TAIL_COM_D = REPULSIVE_CUTOFF_TAIL_COM / LENGTH_SCALE
    REPULSIVE_CUTOFF_TAILS_D = REPULSIVE_CUTOFF_TAILS / LENGTH_SCALE
    COMPLEX_CUTOFF_D = COMPLEX_CUTOFF / LENGTH_SCALE
    INIT_CUTOFF_TAILS_D = INIT_CUTOFF_TAILS / LENGTH_SCALE


    ATTRACTIVE_CUTOFF_D = ATTRACTIVE_CUTOFF / LENGTH_SCALE
    R_C_D = R_C / LENGTH_SCALE
    WCA_REPULSIVE_ENERGY_PARAMETER_D = WCA_REPULSIVE_ENERGY_PARAMETER / ENERGY_SCALE
    TAIL_REPULSIVE_ENERGY_PARAMETER_D = TAIL_REPULSIVE_ENERGY_PARAMETER / ENERGY_SCALE
    ATTRACTIVE_ENERGY_STRENGTH_D = ATTRACTIVE_ENERGY_STRENGTH / ENERGY_SCALE
    SIMULATION_AREA_LENGTH_D = SIMULATION_AREA_LENGTH / LENGTH_SCALE
    print("ATTRACTIVE_CUTOFF_D", ATTRACTIVE_CUTOFF_D)
    print("COMPLEX_CUTOFF_D", COMPLEX_CUTOFF_D)
    print("SIMULATION_AREA_LENGTH_D", SIMULATION_AREA_LENGTH_D)
    print("REPULSIVE_CUTOFF_D", REPULSIVE_CUTOFF_D)
    print("REPULSIVE_CUTOFF_TAILS_D", REPULSIVE_CUTOFF_TAILS_D)
    print("INIT_CUTOFF_TAILS_D", INIT_CUTOFF_TAILS_D)
    # set up the system for simulation

    # ................................random particles......................................
    PARTICLE_NUMBER = PARTNUM
    SIMULATION_STEPS = SIMSTEPS

    MONOMER_CONCENTRTION = PARTICLE_NUMBER / (SIMULATION_AREA_LENGTH_D**2)
    MONOMER_CONCENTRTION_D = PARTICLE_NUMBER / (SIMULATION_AREA_LENGTH_D**2)

    NO_OF_RUNS = 1
    SAVE_INTERVALL = SAVEINT

    params = {'att_c':ATTRACTIVE_CUTOFF_D,
              'rep_c':REPULSIVE_CUTOFF_D,
              'rep_tail_com':REPULSIVE_CUTOFF_TAIL_COM_D,
              'rep_tails':REPULSIVE_CUTOFF_TAILS_D,
              'ang_c':ANGULAR_CUTOFF,
              'rc':R_C_D,
              'comp_c':COMPLEX_CUTOFF_D,
              'init_c':INIT_CUTOFF_TAILS_D,
              'sigma':CHARACTERISTIC_LENGTH_SCALE_POTENTIAL_D,
              'sigma_tails':CHARACTERISTIC_LENGTH_SCALE_POTENTIAL_TAILS_D,
              'sigma_tail_com': CHARACTERISTIC_LENGTH_SCALE_POTENTIAL_TAIL_COM_D,
              'boxlength':SIMULATION_AREA_LENGTH_D,
              'att_ep':ATTRACTIVE_ENERGY_STRENGTH_D,
              'rep_ep':WCA_REPULSIVE_ENERGY_PARAMETER_D,
              'tail_rep_ep':TAIL_REPULSIVE_ENERGY_PARAMETER_D,
              'p1x':PATCH1_X_D,
              'p2x':PATCH2_X_D,
              'p1y':PATCH1_Y_D,
              'p2y':PATCH2_Y_D,
              'pa1x':PATCH1_ALIGNMENT_X_D,
              'pa2x':PATCH2_ALIGNMENT_X_D,
              'pa1y':PATCH1_ALIGNMENT_Y_D,
              'pa2y':PATCH2_ALIGNMENT_Y_D,
              'sp1x':sp1_x,
              'sp2x':sp2_x,
              'sp3x':sp3_x,
              'sp4x':sp4_x,
              'sp5x':sp5_x,
              'sspx':ssp_x,
              'sspy':ssp_y,
              'sp1y':sp1_y,
              'sp2y':sp2_y,
              'sp3y':sp3_y,
              'sp4y':sp4_y,
              'sp5y':sp5_y,
              'tail_x':TAIL_X_D,
              'tail_y':TAIL_Y_D,
              'RBIG':RADIUS_BIG_SPHERE_D,
              'RSMALL':RADIUS_SMALL_SPHERE_D,
              'Dx':DIFFUSION_COEFFICIENT_X_D,
              'Dy':DIFFUSION_COEFFICIENT_Y_D,
              'Dr':DIFFUSION_COEFFICIENT_ROT_D,
              'beta':BETA_D,
              'tstep':TIME_STEP_D,
              'tscale':TIME_SCALE,
              'sdx':STANDARD_DEVIATION_X_D,
              'sdy':STANDARD_DEVIATION_Y_D,
              'sdr':STANDARD_DEVIATION_ROT_D,
              'MAIN_PATH':MAIN_PATH,
              'SUB_PATH':SUB_PATH,
              'simsteps':SIMULATION_STEPS,
              'saveint':SAVE_INTERVALL,
              'cmon':MONOMER_CONCENTRTION_D,
              'savealldata':SAVE_ALL_DATA,
              'tail':TAIL,
              'graphical':GRAPHICAL,
              'only_cluster_statistics':ONLY_STATISTICS,
              'assembly_details':ASSEMBL_DETAILS,
              'tail_angle':ONLY_TAIL_ANGLE,
              'nparticles':PARTICLE_NUMBER,
              'PlanckConstant': h_D,
              'mass' : MASS_D,
              'surface potential': V_SURFACE_D,
              'Reservoir concentration': RHO_RES_D,
              'Montecarlo step': N_m,
              'Trials' : N_trials
    }
    # print("**********Scales**********")
    # print('{:<50s}{:>20s}{:>5E}'.format("LENGTH_SCALE", ' = ', LENGTH_SCALE))
    # print('{:<50s}{:>20s}{:>5E}'.format("TIME_SCALE", ' = ', TIME_SCALE))
    # print('{:<50s}{:>20s}{:>5E}'.format("ENERGY_SCALE", ' = ', ENERGY_SCALE))
    # print('{:<50s}{:>20s}{:>5E}'.format("FORCE_SCALE", ' = ', FORCE_SCALE))
    # print('{:<50s}{:>20s}{:>5E}'.format("TORQUE_SCALE", ' = ', TORQUE_SCALE))
    # print('-' * 100)
    # print("**********Dimensionless Simulation Parameters**********")
    # print('{:<50s}{:>20s}{:>5E}'.format("BETA_D", ' = ', BETA_D))
    # print('{:<50s}{:>20s}{:>5E}'.format("DIFFUSION_COEFFICIENT_X_D", ' = ', DIFFUSION_COEFFICIENT_X_D))
    # print('{:<50s}{:>20s}{:>5E}'.format("DIFFUSION_COEFFICIENT_Y_D", ' = ', DIFFUSION_COEFFICIENT_Y_D))
    # print('{:<50s}{:>20s}{:>5E}'.format("DIFFUSION_COEFFICIENT_ROT_D", ' = ', DIFFUSION_COEFFICIENT_ROT_D))
    # print('{:<50s}{:>20s}{:>5E}'.format("TIME_STEP_D", ' = ', TIME_STEP_D))
    # print('{:<50s}{:>20s}{:>5E}'.format("STANDARD_DEVIATION_X_D", ' = ', STANDARD_DEVIATION_X_D))
    # print('{:<50s}{:>20s}{:>5E}'.format("STANDARD_DEVIATION_Y_D", ' = ', STANDARD_DEVIATION_Y_D))
    # print('{:<50s}{:>20s}{:>5E}'.format("STANDARD_DEVIATION_ROT_D", ' = ', STANDARD_DEVIATION_ROT_D))
    # print('-' * 100)
    # print("**********Dimensionless Geometry (Coordinates in PFF)**********")
    # print('{:<50s}{:>20s}{:>5E}'.format("PATCH1_X_D", ' = ', PATCH1_X_D))
    # print('{:<50s}{:>20s}{:>5E}'.format("PATCH1_Y_D", ' = ', PATCH1_Y_D))
    # print('{:<50s}{:>20s}{:>5E}'.format("PATCH2_X_D", ' = ', PATCH2_X_D))
    # print('{:<50s}{:>20s}{:>5E}'.format("PATCH2_Y_D", ' = ', PATCH2_Y_D))
    # print('{:<50s}{:>20s}{:>5E}'.format("PATCH1_ALIGNMENT_X_D", ' = ', PATCH1_ALIGNMENT_X_D))
    # print('{:<50s}{:>20s}{:>5E}'.format("PATCH1_ALIGNMENT_Y_D", ' = ', PATCH1_ALIGNMENT_Y_D))
    # print('{:<50s}{:>20s}{:>5E}'.format("PATCH2_ALIGNMENT_X_D", ' = ', PATCH2_ALIGNMENT_X_D))
    # print('{:<50s}{:>20s}{:>5E}'.format("PATCH2_ALIGNMENT_Y_D", ' = ', PATCH2_ALIGNMENT_Y_D))
    # print('{:<50s}{:>20s}{:>5E}'.format("RADIUS_BIG_SPHERE_D", ' = ', RADIUS_BIG_SPHERE_D))
    # print('{:<50s}{:>20s}{:>5E}'.format("RADIUS_SMALL_SPHERE_D", ' = ', RADIUS_SMALL_SPHERE_D))
    # print('-' * 100)
    # print("**********Dimensionless Potential Parameters**********")
    # print('{:<50s}{:>20s}{:>5E}'.format("CHARACTERISTIC_LENGTH_SCALE_POTENTIAL_D",
    #                                     ' = ', CHARACTERISTIC_LENGTH_SCALE_POTENTIAL_D))
    # print('{:<50s}{:>20s}{:>5E}'.format("REPULSIVE_CUTOFF_TAILS_D", ' = ', REPULSIVE_CUTOFF_TAILS_D))
    # print('{:<50s}{:>20s}{:>5E}'.format("REPULSIVE_CUTOFF_TAIL_COM_D", ' = ', REPULSIVE_CUTOFF_TAIL_COM_D))
    # print('{:<50s}{:>20s}{:>5E}'.format("REPULSIVE_CUTOFF_D", ' = ', REPULSIVE_CUTOFF_D))
    # print('{:<50s}{:>20s}{:>5E}'.format("ATTRACTIVE_CUTOFF_D", ' = ', ATTRACTIVE_CUTOFF_D))
    # print('{:<50s}{:>20s}{:>5E}'.format("ANGULAR_CUTOFF", ' = ', ANGULAR_CUTOFF))
    # print('{:<50s}{:>20s}{:>5E}'.format("WCA_REPULSIVE_ENERGY_PARAMETER_D",
    #                                     ' = ', WCA_REPULSIVE_ENERGY_PARAMETER_D))
    # print('{:<50s}{:>20s}{:>5E}'.format("ATTRACTIVE_ENERGY_STRENGTH_D", ' = ', ATTRACTIVE_ENERGY_STRENGTH_D))
    # print('{:<50s}{:>20s}{:>5E}'.format("SIMULATION_AREA_LENGTH_D", ' = ', SIMULATION_AREA_LENGTH_D))



    return params
