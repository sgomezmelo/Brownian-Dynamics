import timeit
import numpy as np
from numpy import linalg as LA
import networkx as nx

# count the complexes and decide if the cluster is open or closed
def count_complexes(p12, p21):

    no_particles = p12.shape[0]
    # create container that counts complex size (in addition if closed or not)
    # the index of that list corresponds to the complex size (0 element should always be zero)
    complex_size_open = np.zeros(20)
    complex_size_closed = np.zeros(20)
    # p12 and p21 are matrices which contain only 0 and 1
    # for an undirected graph only the upper triangle (containing the diagonal) is relevant
    p = p12 + p21
    p = np.triu(p)
    # determine indices of entries with 1
    # gives ndarray
    edges = np.argwhere(p==1)
    # transform array to list of tuple
    edges = list(map(tuple, edges))
    # node list of graph (correspnd to particles)
    nodes = list(range(no_particles))
    # create graph and add nodes
    G = nx.Graph()
    G.add_nodes_from(nodes)
    # creating edges (correspond to tuple which gives the indices of entries that contain a 1)
    G.add_edges_from(edges)
    # calculate number of connected components
    # number of conn comp corresponds to number of complexes (at simulation start, the number of
    # conn comp should correspond to the particle number)
    # len(no_complexes) determines the max index for the loop
    no_complexes = nx.number_connected_components(G)
    # transform each connected component to a subgraph in order to check if the complex is closed or not
    graphs = list(G.subgraph(c) for c in nx.connected_components(G))
    # start the loop to count the complex sizes and decide if they are closed or not
    for i in range(no_complexes):
        complex_size = graphs[i].number_of_nodes()
        is_closed = bool(nx.cycle_basis(graphs[i])) # check if the complex is closed

        if is_closed:
            complex_size_closed[complex_size] += 1
        else:
            complex_size_open[complex_size] += 1

    return np.array([complex_size_open, complex_size_closed])

# compute the number of complexes and their size
def comp_complexes(dist_p12, dist_p21, theta12, theta21, params):
    cutoff = params['comp_c']
    theta_cutoff = params['ang_c']
    mask_dist12 = (dist_p12 < cutoff).astype(int)
    mask_dist21 = (dist_p21 < cutoff).astype(int)
    mask_theta12 = (theta12 < theta_cutoff).astype(int)
    mask_theta21 = (theta21 < theta_cutoff).astype(int)

    mask12 = ((mask_dist12 + mask_theta12) == 2).astype(int)
    mask21 = ((mask_dist21 + mask_theta21) == 2).astype(int)

    # array that contains number of certain complex complex sizes
    # complex_sizes[0] for open no_complexes, complex_sizes[1] closed complexes
    complex_sizes = count_complexes(mask12, mask21)

    return complex_sizes


# initalize particles on simulation surface without overlap
def init_particles(params):
    particle_number = params['nparticles']
    area_length = params['boxlength']
    cutoff = params['rep_c']

    x = np.array([0.0] * particle_number)
    y = np.array([0.0] * particle_number)
    first_particle_coord = np.random.uniform(0, area_length, 2)
    positions = np.array([x, y]).T  # index of positions array represents the particle
    positions[0] = first_particle_coord  # first random position assigned to particle 0
    particle = 1
    while particle != particle_number:
        next_coord = np.random.uniform(0, area_length, 2)
        prev_coord = positions[0:particle]
        # print("prev_coord = ",prev_coord)
        difference = prev_coord - next_coord  # calculate difference between existing particle and created particle
        # absolute value of difference between new and each old particle
        abs_difference = LA.norm(difference, axis=1)
        # print("abs_difference = ",abs_difference)
        if min(abs_difference) >= cutoff:  # if the distance of created and each particle is larger than the cutoff,
            positions[particle] = next_coord  # the particle is accepted and the next one is created
            particle = particle + 1
        else:
            pass
    return positions

def init_particles_with_tails(params):
    particle_number = params['nparticles']
    area_length = params['boxlength']
    cutoff = params['init_c']

    x = np.array([0.0] * particle_number)
    y = np.array([0.0] * particle_number)
    first_particle_coord = np.random.uniform(0, area_length, 2)
    positions = np.array([x, y]).T  # index of positions array represents the particle
    positions[0] = first_particle_coord  # first random position assigned to particle 0
    particle = 1
    while particle != particle_number:
        next_coord = np.random.uniform(0, area_length, 2)
        prev_coord = positions[0:particle]
        # print("prev_coord = ",prev_coord)
        difference = prev_coord - next_coord # calculate difference between existing particle and created particle
        difference_periodic = difference - area_length * np.rint(difference / area_length)
        # absolute value of difference between new and each old particle
        abs_difference = LA.norm(difference_periodic, axis=1)
        # print("abs_difference = ",abs_difference)
        if min(abs_difference) >= cutoff:  # if the distance of created and each particle is larger than the cutoff,
            positions[particle] = next_coord  # the particle is accepted and the next one is created
            particle = particle + 1
        else:
            pass
    return positions


#Optimized calculation of differences xi-xj
def compute_dist(coord, params):
    L = params['boxlength']
    d = coord[:,np.newaxis] - coord
   # d[np.abs(d)>L] = d[np.abs(d)>L] - L * np.rint(d[np.abs(d)>L]/L)
    return d

#def compute_dist(coord, params):
    #boxlength = params['boxlength']
    #n = coord.shape[0]
    #matrix = np.empty((n, n))
    #matrix[:] = coord  # fill the whole matrix with coordinate vector (linewise)
    #matrix = matrix.T - matrix  # entries are coord_i - coord_j
    #matrix[:] = (matrix[:] - boxlength * np.rint(matrix[:] / boxlength))
    #return matrix

# returns a matrix with distances dij between body center
def abs_dist_matrix(matrix_x, matrix_y):
    abs_dist_matrix = np.sqrt(matrix_x**2 + matrix_y**2)
    np.fill_diagonal(abs_dist_matrix, 1)
    return abs_dist_matrix


# returns a matrix with distances dij between attractive patches
def compute_dist_patch_AB(coord_patch_A, coord_patch_B, params):
    L = params['boxlength']
    d = coord_patch_A[:,np.newaxis]-coord_patch_B
    np.fill_diagonal(d, 0.0)
   # d[np.abs(d)>L] = d[np.abs(d)>L] - L * np.rint(d[np.abs(d)>L]/L)
    return d


#def compute_dist_patch_AB(coord_patch_A, coord_patch_B, params):
 #   boxlength = params['boxlength']

  #  n = coord_patch_A.shape[0]
   # matrix_A = np.empty((n, n))
   # matrix_B = np.empty((n, n))
   # matrix_A[:] = coord_patch_A
   # matrix_B[:] = coord_patch_B
   # matrix = matrix_A.T - matrix_B
   # matrix[:] = (matrix[:] - boxlength * np.rint(matrix[:] / boxlength))
   # np.fill_diagonal(matrix, 0.0)
    # matrix = matrix - matrix.T
   # return matrix

def compute_dist_tail_com(coord_tail, coord_com, params):
    L = params['boxlength']
    d = coord_tail[:,np.newaxis]-coord_com
    np.fill_diagonal(d, 0.0)
   # d[np.abs(d)>L] = d[np.abs(d)>L] - L * np.rint(d[np.abs(d)>L]/L)
    return d


 #def compute_dist_tail_com(coord_tail, coord_com, params):
    # boxlength = params['boxlength']

     #n = coord_tail.shape[0]
     #matrix_A = np.empty((n, n))
     #matrix_B = np.empty((n, n))
     #matrix_A[:] = coord_tail
     #matrix_B[:] = coord_com
     #matrix = matrix_A.T - matrix_B
     #matrix[:] = (matrix[:] - boxlength * np.rint(matrix[:] / boxlength))
     #np.fill_diagonal(matrix, 0.0)
    # matrix = matrix - matrix.T
     #return matrix

def compute_dist_com_tail(coord_tail, coord_com, params):
    L = params['boxlength']
    d = coord_com[:,np.newaxis]-coord_tail
    np.fill_diagonal(d, 0.0)
   # d[np.abs(d)>L] = d[np.abs(d)>L] - L * np.rint(d[np.abs(d)>L]/L)
    return d


# computes theta_AB_ij angle between patch A and B for each combination of i and j but i != j
def compute_angle_patch_AB(align_patch_A_x, align_patch_A_y, align_patch_B_x, align_patch_B_y):
    dot_product = align_patch_A_x[:,np.newaxis]*align_patch_B_x +  align_patch_A_y[:,np.newaxis]*align_patch_B_y
    np.fill_diagonal(dot_product, 0)
    return np.arccos(-dot_product)

# def compute_angle_patch_AB(align_patch_A_x, align_patch_A_y, align_patch_B_x, align_patch_B_y):
   #  n = align_patch_A_x.shape[0]
    # matrix_A_x = np.empty((n, n))
    # matrix_B_x = np.empty((n, n))
    # matrix_A_y = np.empty((n, n))
    # matrix_B_y = np.empty((n, n))
    # fill matrices
    # matrix_A_x[:] = align_patch_A_x
    # matrix_B_x[:] = align_patch_B_x
    # matrix_A_y[:] = align_patch_A_y
    # matrix_B_y[:] = align_patch_B_y
    # compute the scalar product using matrix elementwise matrix multilpication
    # matrix_x = matrix_A_x.T * matrix_B_x
    # matrix_y = matrix_A_y.T * matrix_B_y
    # add matrices together
    # matrix = matrix_x + matrix_y
    # np.fill_diagonal(matrix, 0)
    # theta_AB = np.arccos(-matrix)

    return theta_AB


def total_force(force_matrix):
    force = np.sum(force_matrix, axis=1)
    return force


def initalize_particle_orientation(angular_range, params):
    ''' Initalize particle orientation.
        For each particle create and angle "alpha" which describes its inital orientation with repsect to
        the lab frame.

        input:
        angular_range: float
            range from which the angle is drawn (maximal range [0, 2PI])
        particle_number: int
            number of solute particles
    '''
    particle_number = params['nparticles']
    angle = np.array(np.random.uniform(0, angular_range, particle_number))
    return angle


def patch1_x_lab(x, y, cosinus_alpha, sinus_alpha, params):
    ''' Transform x-coordinate of patch1 to the lab frame.
        In each simulation step the orientation of the particles change.

    '''
    PATCH1_X_D = params['p1x']
    PATCH1_Y_D = params['p1y']
    patch_x_lab = x + cosinus_alpha * PATCH1_X_D - sinus_alpha * PATCH1_Y_D
    return patch_x_lab


def patch1_y_lab(x, y, cosinus_alpha, sinus_alpha, params):
    PATCH1_X_D = params['p1x']
    PATCH1_Y_D = params['p1y']
    patch_y_lab = y + sinus_alpha * PATCH1_X_D + cosinus_alpha * PATCH1_Y_D
    return patch_y_lab


def patch2_x_lab(x, y, cosinus_alpha, sinus_alpha, params):
    PATCH2_X_D = params['p2x']
    PATCH2_Y_D = params['p2y']
    patch_x_lab = x + cosinus_alpha * PATCH2_X_D - sinus_alpha * PATCH2_Y_D
    return patch_x_lab


def patch2_y_lab(x, y, cosinus_alpha, sinus_alpha, params):
    PATCH2_X_D = params['p2x']
    PATCH2_Y_D = params['p2y']
    patch_y_lab = y + sinus_alpha * PATCH2_X_D + cosinus_alpha * PATCH2_Y_D
    return patch_y_lab


def patch1_alignment_x_lab(cosinus_alpha, sinus_alpha, params):
    PATCH1_ALIGNMENT_X_D = params['pa1x']
    PATCH1_ALIGNMENT_Y_D = params['pa1y']
    patch_alignment_x_lab = cosinus_alpha * PATCH1_ALIGNMENT_X_D - sinus_alpha * PATCH1_ALIGNMENT_Y_D
    return patch_alignment_x_lab


def patch1_alignment_y_lab(cosinus_alpha, sinus_alpha, params):
    PATCH1_ALIGNMENT_X_D = params['pa1x']
    PATCH1_ALIGNMENT_Y_D = params['pa1y']
    patch_alignment_y_lab = sinus_alpha * PATCH1_ALIGNMENT_X_D + cosinus_alpha * PATCH1_ALIGNMENT_Y_D
    return patch_alignment_y_lab


def patch2_alignment_x_lab(cosinus_alpha, sinus_alpha, params):
    PATCH2_ALIGNMENT_X_D = params['pa2x']
    PATCH2_ALIGNMENT_Y_D = params['pa2y']
    patch_alignment_x_lab = cosinus_alpha * PATCH2_ALIGNMENT_X_D - sinus_alpha * PATCH2_ALIGNMENT_Y_D
    return patch_alignment_x_lab


def patch2_alignment_y_lab(cosinus_alpha, sinus_alpha, params):
    PATCH2_ALIGNMENT_X_D = params['pa2x']
    PATCH2_ALIGNMENT_Y_D = params['pa2y']
    patch_alignment_y_lab = sinus_alpha * PATCH2_ALIGNMENT_X_D + cosinus_alpha * PATCH2_ALIGNMENT_Y_D
    return patch_alignment_y_lab

# torque in pff check theory again


def torque_in_pff(attr_force_x, attr_force_y, cosinus_alpha, sinus_alpha, patch_coord_x, patch_coord_y):
    ''' Calculate the torque on the particle due to the forces acting on a specific patch.

    '''
    force_PFF_x = cosinus_alpha * attr_force_x + sinus_alpha * attr_force_y
    force_PFF_y = -sinus_alpha * attr_force_x + cosinus_alpha * attr_force_y

    torque_on_patch = patch_coord_x * force_PFF_y - patch_coord_y * force_PFF_x

    return torque_on_patch


def force_x_pff(force_x, force_y, cosinus_alpha, sinus_alpha):
    ''' Transform the force in x direction into the particle fixed frame (PFF)

    '''
    force_x_pff = cosinus_alpha * force_x + sinus_alpha * force_y
    return force_x_pff


def force_y_pff(force_x, force_y, cosinus_alpha, sinus_alpha):
    ''' Transform the force in y direction into the particle fixed frame (PFF)

    '''
    force_y_pff = -sinus_alpha * force_x + cosinus_alpha * force_y
    return force_y_pff

def spheres_x_lab(x, y, cosinus_alpha, sinus_alpha, params):
    ''' Transform x-coordinate of patch1 to the lab frame.
        In each simulation step the orientation of the particles change.

    '''
    sp1_x = params['sp1x']
    sp2_x = params['sp2x']
    sp3_x = params['sp3x']
    sp4_x = params['sp4x']
    sp5_x = params['sp5x']

    sp1_y = params['sp1y']
    sp2_y = params['sp2y']
    sp3_y = params['sp3y']
    sp4_y = params['sp4y']
    sp5_y = params['sp5y']

    sp1_x_lab = x + cosinus_alpha * sp1_x - sinus_alpha * sp1_y
    sp2_x_lab = x + cosinus_alpha * sp2_x - sinus_alpha * sp2_y
    sp3_x_lab = x + cosinus_alpha * sp3_x - sinus_alpha * sp3_y
    sp4_x_lab = x + cosinus_alpha * sp4_x - sinus_alpha * sp4_y
    sp5_x_lab = x + cosinus_alpha * sp5_x - sinus_alpha * sp5_y


    return sp1_x_lab, sp2_x_lab, sp3_x_lab, sp4_x_lab, sp5_x_lab

def ssp_x_lab(x, y, cosinus_alpha, sinus_alpha, params):
    ''' Transform x-coordinate of patch1 to the lab frame.
        In each simulation step the orientation of the particles change.

    '''
    ssp_x = params['sspx']
    ssp_y = params['sspy']

    ssp_x_lab = x + cosinus_alpha * ssp_x - sinus_alpha * ssp_y

    return ssp_x_lab

def spheres_y_lab(x, y, cosinus_alpha, sinus_alpha, params):
    ''' Transform x-coordinate of patch1 to the lab frame.
        In each simulation step the orientation of the particles change.

    '''
    sp1_x = params['sp1x']
    sp2_x = params['sp2x']
    sp3_x = params['sp3x']
    sp4_x = params['sp4x']
    sp5_x = params['sp5x']

    sp1_y = params['sp1y']
    sp2_y = params['sp2y']
    sp3_y = params['sp3y']
    sp4_y = params['sp4y']
    sp5_y = params['sp5y']

    sp1_y_lab = y + sinus_alpha * sp1_x + cosinus_alpha * sp1_y
    sp2_y_lab = y + sinus_alpha * sp2_x + cosinus_alpha * sp2_y
    sp3_y_lab = y + sinus_alpha * sp3_x + cosinus_alpha * sp3_y
    sp4_y_lab = y + sinus_alpha * sp4_x + cosinus_alpha * sp4_y
    sp5_y_lab = y + sinus_alpha * sp5_x + cosinus_alpha * sp5_y


    return sp1_y_lab, sp2_y_lab, sp3_y_lab, sp4_y_lab, sp5_y_lab

def ssp_y_lab(x, y, cosinus_alpha, sinus_alpha, params):
    ''' Transform x-coordinate of patch1 to the lab frame.
        In each simulation step the orientation of the particles change.

    '''
    ssp_x = params['sspx']
    ssp_y = params['sspy']

    ssp_y_lab = y + sinus_alpha * ssp_x + cosinus_alpha * ssp_y

    return ssp_y_lab

def tail_x_lab(cosinus_alpha, sinus_alpha, params):
    ''' Transform x-coordinate of tail to the lab frame.
        In each simulation step the orientation of the particles change.

    '''
    TAIL_X_D = params['tail_x']
    TAIL_Y_D = params['tail_y']
    tail_x_lab = cosinus_alpha * TAIL_X_D - sinus_alpha * TAIL_Y_D
    return tail_x_lab

def tail_y_lab(cosinus_alpha, sinus_alpha, params):
    ''' Transform x-coordinate of tail to the lab frame.
        In each simulation step the orientation of the particles change.

    '''
    TAIL_X_D = params['tail_x']
    TAIL_Y_D = params['tail_y']
    tail_y_lab = sinus_alpha * TAIL_X_D + cosinus_alpha * TAIL_Y_D
    return tail_y_lab

# Attractive and repulsive potentials for Montecarlo Simulation of open system

def att_potential(distance, angle, params):
    e_att = params['att_ep']
    sigma = params['sigma']
    cutoff_att = params['att_c']
    cutoff_angle = params['ang_c']
    rc = params['rc']

    U = np.zeros(distance.shape)
    #Non zero potential will be by d<cutoff and angle<cutoff
    non_zero =  np.logical_and(distance<=cutoff_att,angle<=cutoff_angle)

    d = distance[non_zero]
    theta = angle[non_zero]
    

    u_radial = 4*e_att*(((sigma)/(d+2**(1/6)*sigma))**12-((sigma)/(d+2**(1/6)*sigma))**6 - (sigma/rc)**12 + (sigma/rc)**6)
    u_angular = 0.5*(np.cos(np.pi*theta/cutoff_angle)+1)

    U[non_zero] = u_radial*u_angular

    return U

def rep_potential(distance, params, tails=False, tailscom=False):

    if tails is True:
        epsilon_rep = params['tail_rep_ep']
        sigma = params['sigma_tails']
        cutoff_rep = params['rep_tails']

    if tailscom is True:

        epsilon_rep = params['tail_rep_ep']
        sigma = params['sigma_tail_com']
        cutoff_rep = params['rep_tail_com']

    if (tails is False and tailscom is False):

        epsilon_rep = params['rep_ep']
        sigma = params['sigma']
        cutoff_rep = params['rep_c']

    U = np.zeros(len(distance))
    d = distance[distance <= cutoff_rep]
    U[distance<=cutoff_rep] = 4*epsilon_rep*((sigma/d)**12-(sigma/d)**6)+epsilon_rep

    return U


#Probability of creation/annhiliation of a new particle from/to reservoir

def create_montecarlo(d,d_patches_1,d_patches_2,theta_1,theta_2,params):
    #By definition dimensionless
    L = params['boxlength']
    beta = params['beta'] 
    h = params['PlanckConstant']
    m = params['mass']
    Vo = params['surface potential']
    rho_res = params['Reservoir concentration']
    
    A = L**2
    N  = len(d)
    Lambda = h/np.sqrt(2*np.pi*m/beta)

    U_rep = np.sum(rep_potential(d,params))
    U_att = np.sum(att_potential(d_patches_1,theta_1,params)+att_potential(d_patches_2,theta_2,params))
    
    E = U_rep+U_att-Vo
    p =  (rho_res*A*Lambda/(N+1))*np.exp(-beta*E)
    p_create = p/(1+p)

    return p_create

def destroy_montecarlo(d,d_patches_1,d_patches_2,theta_1,theta_2,params):
     #By definition dimensionless
    L = params['boxlength']
    beta = params['beta'] 
    h = params['PlanckConstant']
    m = params['mass']
    Vo = params['surface potential']
    rho_res = params['Reservoir concentration']
    
    A = L**2
    N  = len(d)
    Lambda = h/np.sqrt(2*np.pi*m/beta)

    U_rep = np.sum(rep_potential(d,params))
    U_att = np.sum(att_potential(d_patches_1,theta_1,params)+att_potential(d_patches_2,theta_2,params))
    
    E = (U_rep+U_att-Vo)
    print('E',E)
    p =  (rho_res*A*Lambda/(N))*np.exp(-beta*E)

    p_destroy = 1/(1+p)

    return p_destroy
    


#N = 100
#L  = 100
#trial  = np.random.uniform(size = N)*L
#trial_2 =  np.random.uniform(size = N)*L
#params = {'boxlength': 100}
#fast_dist(trial,trial_2,params)
#'''

# test2 = '''
#N = 100
#L  = 100
#trial  = np.random.uniform(size = N)*L
#trial_2 =  np.random.uniform(size = N)*L
#params = {'boxlength': 100}
#compute_dist_com_tail(trial,trial_2,params)
#'''
#print('Newaxis')
#print(timeit.timeit(test1,'from __main__ import fast_dist, np', number = 10)/10)
#print('Dennis')
#print(timeit.timeit(test2,'from __main__ import compute_dist_com_tail, np', number = 10)/10)
##
