import numpy as np
from numpy import linalg as LA
import networkx as nx
import matplotlib.pyplot as plt
import time


#Counts the number of complexes and labels them as open or closed
#Takes as input matrices p12 and p21 where (p12)[i,j] = 1 if the patches 1 and 2 of particles i and j are bound, and 0 otherwise.
#Returns a numpy array where the entry [j,0] corresponds to the number of open j-mers and [j,1] to the number of closed j-mers.
def count_complexes(p12, p21):

    no_particles = p12.shape[0] #Extract number of monomers.
    N_max  = 20 #Neglect any complex size with more than 20 monomers.
    
    complex_size_open = np.zeros(N_max) # Create container to count open complexes. The number of j-mers will be stored at entry j.
    complex_size_closed = np.zeros(N_max) #Ccreate container to count closed complexes. The number of j-mers will be stored at entry j.

    p = p12 + p21 #Extract which monomers are bound together, regardless of the patch label.
    p = np.triu(p) # Exctract upper triangular half ofp, which is a symmetric matrix because graph is undirected

    edges = np.argwhere(p==1) #Numpy array with indices where p matrix is 1.
    edges = list(map(tuple, edges))# Transform array to list of tuple
   
    nodes = list(range(no_particles)) # node list of graph (correspnd to particles)
    # Create graph and add nodes
    G = nx.Graph()
    G.add_nodes_from(nodes)
    # Create edges (correspond to tuple which gives the indices of entries that contain a 1)
    G.add_edges_from(edges)
    # Calculate number of connected components, corresponding to the number of complexes
    no_complexes = nx.number_connected_components(G)
    # Transform each connected component to a subgraph in order to check if the complex is closed or not
    graphs = list(G.subgraph(c) for c in nx.connected_components(G))
    #Loop to count the complex sizes and decide if they are closed or not
    for i in range(no_complexes):
        complex_size = graphs[i].number_of_nodes()
        is_closed = bool(nx.cycle_basis(graphs[i])) # Check if the complex is closed

        if complex_size < N_max:
            if is_closed:
                complex_size_closed[complex_size] += 1
            else:
                complex_size_open[complex_size] += 1
    return np.array([complex_size_open, complex_size_closed])



# Takes as input four NxN matrices, as well as parameters in params
# Matrices dist_p12 and dist_p21 have the distances between pairs of patches
# Matrices theta12 and theta21 have angular orienations between pairs of patches
# Returns array where entry [j,0] corresponds to the number of open j-mers and [j,1] to the number of closed j-mers.
def comp_complexes(dist_p12, dist_p21, theta12, theta21, params):
    #Extract distance and angular cutoffs for bond.
    cutoff = params['comp_c']
    theta_cutoff = params['ang_c']
    
    #Determine which patches are closer than the cutoff bond distance
    mask_dist12 = (dist_p12 < cutoff).astype(int)
    mask_dist21 = (dist_p21 < cutoff).astype(int)
    
    #Determine which patches have narrower relative orientation than angle cutodd
    mask_theta12 = (theta12 < theta_cutoff).astype(int)
    mask_theta21 = (theta21 < theta_cutoff).astype(int)

    #Determine which pairs patches fulfill both distance and angle criteria
    mask12 = ((mask_dist12 + mask_theta12) == 2).astype(int)
    mask21 = ((mask_dist21 + mask_theta21) == 2).astype(int)

    #Count open and closed complexes
    complex_sizes = count_complexes(mask12, mask21)

    return complex_sizes


# Initalize particles on simulation surface without overlap
def init_particles(params):
    #Extract relevant parameters
    particle_number = params['nparticles']
    area_length = params['boxlength']
    cutoff = params['rep_c']
    
    #Initialize arrays
    x = np.array([0.0] * particle_number)
    y = np.array([0.0] * particle_number)
    first_particle_coord = np.random.uniform(0, area_length, 2)
    positions = np.array([x, y]).T  # index of positions array represents the particle
    positions[0] = first_particle_coord  # first random position assigned to particle 0
    particle = 1
    while particle != particle_number:
        next_coord = np.random.uniform(0, area_length, 2)
        prev_coord = positions[0:particle]
        difference = prev_coord - next_coord  # calculate difference between existing particle and created particle
        # absolute value of difference between new and each old particle
        abs_difference = LA.norm(difference, axis=1)
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
    
    #Initialize empty array for positions 
    x = np.array([0.0] * particle_number)
    y = np.array([0.0] * particle_number)
    first_particle_coord = np.random.uniform(0, area_length, 2)
    positions = np.array([x, y]).T  
    positions[0] = first_particle_coord  # first random position assigned to particle 0
    particle = 1
    while particle != particle_number:
        next_coord = np.random.uniform(0, area_length, 2)
        prev_coord = positions[0:particle]
        difference = prev_coord - next_coord # calculate difference between existing particle and created particle
        difference_periodic = difference - area_length * np.rint(difference / area_length)
        # absolute value of difference between new and each old particle
        abs_difference = LA.norm(difference_periodic, axis=1)
        if min(abs_difference) >= cutoff:  # if the distance of created and each particle is larger than the cutoff,
            positions[particle] = next_coord  # the particle is accepted and the next one is created
            particle = particle + 1
        else:
            pass
    return positions


#Calculate differences in coordinates Xj-Xi of body centers according to minimum distance convention
def compute_dist(coord, params):
    L = params['boxlength']
    d = coord[:,np.newaxis] - coord #Compute Xj-Xi
    d[np.abs(d)>L/2.0] = d[np.abs(d)>L/2.0] - L*np.sign(d[np.abs(d)>L/2.0]) #Minimum distance convention
    return d

# Returns a matrix with distances dij between body centers
def abs_dist_matrix(matrix_x, matrix_y):
    abs_dist_matrix = np.sqrt(matrix_x**2 + matrix_y**2)
    np.fill_diagonal(abs_dist_matrix, 1)
    return abs_dist_matrix


# Returns a matrix with distances dij between attractive patches
def compute_dist_patch_AB(coord_patch_A, coord_patch_B, params):
    L = params['boxlength']
    d = coord_patch_A[:,np.newaxis]-coord_patch_B
    np.fill_diagonal(d, 0.0)
    d[np.abs(d)>L/2.0] =   d[np.abs(d)>L/2.0] - L*np.sign(d[np.abs(d)>L/2.0])
    return d

#Calculate differences in coordinates Xj-Xi of body center and tails according to minimum distance convention
def compute_dist_tail_com(coord_tail, coord_com, params):
    L = params['boxlength']
    d = coord_tail[:,np.newaxis]-coord_com
    d[np.abs(d)>L/2.0] = d[np.abs(d)>L/2.0] -  L*np.sign(d[np.abs(d)>L/2.0])
    return d

#Calculate differences in coordinates Xj-Xi of body centers according to minimum distance conventio
def compute_dist_com_tail(coord_tail, coord_com, params):
    L = params['boxlength']
    d = coord_com[:,np.newaxis]-coord_tail
    d[np.abs(d)>L/2.0] = d[np.abs(d)>L/2.0] - L*np.sign(d[np.abs(d)>L/2.0])
    return d


# Computes theta_AB_ij angle between patch A and B for each combination of i and j but i != j
def compute_angle_patch_AB(align_patch_A_x, align_patch_A_y, align_patch_B_x, align_patch_B_y):
    dot_product = align_patch_A_x[:,np.newaxis]*align_patch_B_x +  align_patch_A_y[:,np.newaxis]*align_patch_B_y
    
    #Alignment is normalized by construction, so anything bigger than 1 is a float point error
    dot_product[dot_product>1.0] = 1.0
    dot_product[dot_product<-1.0] = -1.0
    theta = np.arccos(-dot_product)
    np.fill_diagonal(theta, 0)
    return theta


# Adds all forces over an axis
def total_force(force_matrix):
    force = np.sum(force_matrix, axis=1)
    return force

#Randomly initializes particles orientation
#Takes as input the maximum range, so that the orientations are in the interval [0, angular_range]
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

#Calculate x coordinate of patch + in the lab frame
def patch1_x_lab(x, y, cosinus_alpha, sinus_alpha, params):
    ''' Transform x-coordinate of patch1 to the lab frame.
        In each simulation step the orientation of the particles change.

    '''
    PATCH1_X_D = params['p1x']
    PATCH1_Y_D = params['p1y']
    patch_x_lab = x + cosinus_alpha * PATCH1_X_D - sinus_alpha * PATCH1_Y_D
    return patch_x_lab

#Calculate y coordinate of patch + in the lab frame
def patch1_y_lab(x, y, cosinus_alpha, sinus_alpha, params):
    PATCH1_X_D = params['p1x']
    PATCH1_Y_D = params['p1y']
    patch_y_lab = y + sinus_alpha * PATCH1_X_D + cosinus_alpha * PATCH1_Y_D
    return patch_y_lab

#Calculate x coordinate of patch - in the lab frame
def patch2_x_lab(x, y, cosinus_alpha, sinus_alpha, params):
    PATCH2_X_D = params['p2x']
    PATCH2_Y_D = params['p2y']
    patch_x_lab = x + cosinus_alpha * PATCH2_X_D - sinus_alpha * PATCH2_Y_D
    return patch_x_lab

#Calculate y coordinate of patch - in the lab frame
def patch2_y_lab(x, y, cosinus_alpha, sinus_alpha, params):
    PATCH2_X_D = params['p2x']
    PATCH2_Y_D = params['p2y']
    patch_y_lab = y + sinus_alpha * PATCH2_X_D + cosinus_alpha * PATCH2_Y_D
    return patch_y_lab

#Rotates the x component of the + patch vector px to the lab frame
def patch1_alignment_x_lab(cosinus_alpha, sinus_alpha, params):
    PATCH1_ALIGNMENT_X_D = params['pa1x']
    PATCH1_ALIGNMENT_Y_D = params['pa1y']
    patch_alignment_x_lab = cosinus_alpha * PATCH1_ALIGNMENT_X_D - sinus_alpha * PATCH1_ALIGNMENT_Y_D
    return patch_alignment_x_lab

#Rotates the y component of the + patch vector px to the lab frame
def patch1_alignment_y_lab(cosinus_alpha, sinus_alpha, params):
    PATCH1_ALIGNMENT_X_D = params['pa1x']
    PATCH1_ALIGNMENT_Y_D = params['pa1y']
    patch_alignment_y_lab = sinus_alpha * PATCH1_ALIGNMENT_X_D + cosinus_alpha * PATCH1_ALIGNMENT_Y_D
    return patch_alignment_y_lab

#Rotates the x component of the - patch vector px to the lab frame
def patch2_alignment_x_lab(cosinus_alpha, sinus_alpha, params):
    PATCH2_ALIGNMENT_X_D = params['pa2x']
    PATCH2_ALIGNMENT_Y_D = params['pa2y']
    patch_alignment_x_lab = cosinus_alpha * PATCH2_ALIGNMENT_X_D - sinus_alpha * PATCH2_ALIGNMENT_Y_D
    return patch_alignment_x_lab

#Rotates the y component of the - patch vector px to the lab frame
def patch2_alignment_y_lab(cosinus_alpha, sinus_alpha, params):
    PATCH2_ALIGNMENT_X_D = params['pa2x']
    PATCH2_ALIGNMENT_Y_D = params['pa2y']
    patch_alignment_y_lab = sinus_alpha * PATCH2_ALIGNMENT_X_D + cosinus_alpha * PATCH2_ALIGNMENT_Y_D
    return patch_alignment_y_lab

#Calculates the cross product rxF of torques. Since F and r are on the xy plane, 
#Torques are purely along the z axis (out of plane)
def torque_in_pff(attr_force_x, attr_force_y, cosinus_alpha, sinus_alpha, patch_coord_x, patch_coord_y):
    ''' Calculate the torque on the particle due to the forces acting on a specific patch.

    '''
    force_PFF_x = cosinus_alpha * attr_force_x + sinus_alpha * attr_force_y
    force_PFF_y = -sinus_alpha * attr_force_x + cosinus_alpha * attr_force_y

    torque_on_patch = patch_coord_x * force_PFF_y - patch_coord_y * force_PFF_x

    return torque_on_patch

#Rotates the x component of the force to the particle fixed frame
def force_x_pff(force_x, force_y, cosinus_alpha, sinus_alpha):
    ''' Transform the force in x direction into the particle fixed frame (PFF)

    '''
    force_x_pff = cosinus_alpha * force_x + sinus_alpha * force_y
    return force_x_pff

#Rotates the y component of the force to the particle fixed frame
def force_y_pff(force_x, force_y, cosinus_alpha, sinus_alpha):
    ''' Transform the force in y direction into the particle fixed frame (PFF)

    '''
    force_y_pff = -sinus_alpha * force_x + cosinus_alpha * force_y
    return force_y_pff

# Given the coordinates of the particle, calculate the x coordinates of the 
# five spheres representing the SAS-6 coarse model in the lab frame. 
def spheres_x_lab(x, y, cosinus_alpha, sinus_alpha, params):
    ''' Returns the x coordinates of the five sphere model of the protein in 
        the lab frame
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

#X-coordinate of the patch in the lab frame, for graphing purposes
def ssp_x_lab(x, y, cosinus_alpha, sinus_alpha, params):
    ''' Transform x-coordinate of patch1 to the lab frame.
        In each simulation step the orientation of the particles change.

    '''
    ssp_x = params['sspx']
    ssp_y = params['sspy']

    ssp_x_lab = x + cosinus_alpha * ssp_x - sinus_alpha * ssp_y

    return ssp_x_lab

# Given the coordinates of the particle, calculate the y coordinates of the 
# five spheres representing the SAS-6 coarse model in the lab frame. 
def spheres_y_lab(x, y, cosinus_alpha, sinus_alpha, params):
    ''' Returns the y coordinates of the five sphere model of the protein in 
        the lab frame
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

#Y-coordinate of the patch in the lab frame, for graphing purposes
def ssp_y_lab(x, y, cosinus_alpha, sinus_alpha, params):
    ''' Transform y-coordinate of patch1 to the lab frame.
        In each simulation step the orientation of the particles change.

    '''
    ssp_x = params['sspx']
    ssp_y = params['sspy']

    ssp_y_lab = y + sinus_alpha * ssp_x + cosinus_alpha * ssp_y

    return ssp_y_lab

#Transforms x coordinate of the tail to the lab frame
def tail_x_lab(cosinus_alpha, sinus_alpha, params):
    ''' Transform x-coordinate of tail to the lab frame.
        In each simulation step the orientation of the particles change.

    '''
    TAIL_X_D = params['tail_x']
    TAIL_Y_D = params['tail_y']
    tail_x_lab = cosinus_alpha * TAIL_X_D - sinus_alpha * TAIL_Y_D
    return tail_x_lab

#Transforms y coordinate of the tail to the lab frame
def tail_y_lab(cosinus_alpha, sinus_alpha, params):
    ''' Transform y-coordinate of tail to the lab frame.
        In each simulation step the orientation of the particles change.

    '''
    TAIL_X_D = params['tail_x']
    TAIL_Y_D = params['tail_y']
    tail_y_lab = sinus_alpha * TAIL_X_D + cosinus_alpha * TAIL_Y_D
    return tail_y_lab

# Calculation of the attractive potential from pairwise patch distances and angles
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

#Calculates pairwise repulsive potentials based on distances. 
def rep_potential(distance, params, tails=False, tailscom=False):
    
    #Determine if it is a body-body, tail-tail or body-tail interaction
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
        
    #Prepare empty container and determine which distances are within cutoff
    U = np.zeros(len(distance))
    d = distance[distance <= cutoff_rep]
    
    #Calculate non zero components of the repuslive potential.
    U[distance<=cutoff_rep] = 4*epsilon_rep*((sigma/d)**12-(sigma/d)**6)+epsilon_rep

    return U


#Calculate the probability of accepting an adsorption event of a test particle
def create_montecarlo(d,d_patches_1,d_patches_2,theta_1,theta_2,params, d_tails,d_tailscom, d_comtails):
    #Extract relevant parameters
    L = params['boxlength']
    beta = params['beta']
    h = params['PlanckConstant']
    m = params['mass']
    Vo = params['surface potential']
    rho_res = params['Reservoir concentration']
    tails = params['tail']
    cutoff_com = params['rep_c']
    cutoff_tail = params['rep_tails']
    attraction = params['Attraction GCMC']
    s_exclusion = params['Exclusion Surface']
    #Define cutoffs to avoid overflowng errors - probability is set to 0 is distances are less than these cutoffs
    cutoff_MC = params['sigma']*0.5
    cutoff_MC_tail = params['sigma_tails']*0.5
    
    A = L**2
    N  = len(d)
    Lambda = h/np.sqrt(2*np.pi*m/beta)
    
    #Surface exclusion - No overlapping between particles at all (Hard sphere adsorption)
    if s_exclusion:
        
        #Determine if any range of interaction of two particles overlap
        surface_exclusion = np.any(d<cutoff_com)
        #Consider tail overlap 
        if tails == True:
            surface_exclusion_tcom = np.any(d_tailscom < cutoff_com)
            surface_exclusion_tt = np.any(d_tails < cutoff_tail)
            surface_exclusion_comt = np.any(d_comtails < cutoff_com)
            surface_exclusion = surface_exclusion = (surface_exclusion_tt | surface_exclusion_tcom) | (surface_exclusion | surface_exclusion_comt)
        if surface_exclusion:
            #If they do overlap, reject this move. 
            return 0.0
        else:
            #Calculate probability solely from surface electrostatic interaction V0
            if attraction == False:
                E = -Vo
                p =  (rho_res*A*Lambda/float(N+1))*np.exp(-beta*E)
                return np.min([p,1.0])
            
            #Consider contribution from attractive interactions
            else:
                U_att = np.sum(att_potential(d_patches_1,theta_1,params)+att_potential(d_patches_2,theta_2,params))
                E = U_att-Vo
                p =  (rho_res*A*Lambda/float(N+1))*np.exp(-beta*E)
                return np.min([p,1.0])
        
    #Consider actual repulsive potentials if there is overlap
    else:
        #Reject move if the overlap is significant (avoids overflowing errors from V = 1/r)
        if np.any(d<cutoff_MC):
            return 0.0
        else:
            #Sum repulsive potentials 
            U_rep = np.sum(rep_potential(d,params))
            if tails:
                #See if the overlap of tails is less than the MC cutoff
                tail_overlap = np.any(d_tails < cutoff_MC_tail)
                tailcom_overlap = np.any(d_tailscom < cutoff_MC_tail)
                comtail_overlap = np.any(d_comtails < cutoff_MC_tail)
                overlap = (tail_overlap|tailcom_overlap)|comtail_overlap
                
                #Rejects move if overlap is less than MC cutoff (avoids overflow errors)
                if overlap:
                    return 0.0
                else:
                    #Computes additional repulsive interactions due to tails
                    U_tails_tails = np.sum(rep_potential(d_tails,params, tails = True))
                    #print('tt',U_tails_tails)
                    U_tails_com = np.sum(rep_potential(d_tailscom,params, tailscom = True))
                    #print('tc',U_tails_com)
                    U_com_tails = np.sum(rep_potential(d_comtails,params, tailscom = True))
                    #print('tc',U_com_tails)
                    U_rep = U_rep + U_tails_tails + U_tails_com + U_com_tails
    
    
            #Calculate total potential Energy as particle interaction + substrate interaction
            if attraction == False:
                E = U_rep-Vo
            #Include attractive contribution if estipulated by user
            else:
                U_att = np.sum(att_potential(d_patches_1,theta_1,params)+att_potential(d_patches_2,theta_2,params))
                E = U_rep+U_att-Vo
            
            #Calculate acceptance probability using Metropolis Hastings
            p =  (rho_res*A*Lambda/float(N+1))*np.exp(-beta*E)
            return np.min([p,1.0])


#Calculate the probability of accepting a desorption event of a test particle
def destroy_montecarlo(d,d_patches_1,d_patches_2,theta_1,theta_2,params, d_tails, d_tailscom, d_comtails):
     #By definition dimensionless
    L = params['boxlength']
    beta = params['beta']
    h = params['PlanckConstant']
    m = params['mass']
    Vo = params['surface potential']
    rho_res = params['Reservoir concentration']
    tails = params['tail']
    attraction = params['Attraction GCMC']
    s_exclusion = params['Exclusion Surface']
    cutoff_com = params['rep_c']
    cutoff_tail = params['rep_tails']
    
    A = L**2
    N  = len(d)
    Lambda = h/np.sqrt(2*np.pi*m/beta)
    
    #Surface exclusion - No overlapping between particles at all (Hard sphere adsorption)
    if s_exclusion:
        #Determine if any range of interaction of two particles overlap
        surface_exclusion = np.any(d<cutoff_com)
        if tails == True:
            #See if the overlap of tails is less than the MC cutoff
            surface_exclusion_tcom = np.any(d_tailscom < cutoff_com)
            surface_exclusion_tt = np.any(d_tails < cutoff_tail)
            surface_exclusion_comt = np.any(d_comtails < cutoff_com)
            surface_exclusion = surface_exclusion = (surface_exclusion_tt | surface_exclusion_tcom) | (surface_exclusion | surface_exclusion_comt)
        if surface_exclusion:
            #Accept move if overlapping occurs
            return 1.0
        else:
            #Compute probability solely based on surface potential V0
            if attraction == False:
                E = -Vo
                p =  (float(N)/rho_res*A*Lambda)*np.exp(beta*E)
                return np.min([p,1.0])
            #Add interaction potential if specified by the user
            else:
                U_att = np.sum(att_potential(d_patches_1,theta_1,params)+att_potential(d_patches_2,theta_2,params))
                E = U_att-Vo
                p =  (float(N)/rho_res*A*Lambda)*np.exp(beta*E)
                return np.min([p,1.0])
    
    else:
        #Calculate repulsive potential due to all particles
        U_rep = np.sum(rep_potential(d,params))
        
        #Calculate Tail-Tail and Body-tail repulsive interaction when tails are taken into account
        if tails == True:
            U_tails_tails = np.sum(rep_potential(d_tails,params, tails = True))
            U_tails_com = np.sum(rep_potential(d_tailscom,params, tailscom = True))
            U_com_tails = np.sum(rep_potential(d_comtails,params, tailscom = True))
            U_rep = U_rep + U_tails_tails + U_tails_com + U_com_tails
            
        #Compute total energy if attractive energy is neglected
        if attraction == False:
            E = U_rep-Vo
        #Add contribution from attractive potentials
        else:
            U_att = np.sum(att_potential(d_patches_1,theta_1,params)+att_potential(d_patches_2,theta_2,params))
            E = U_rep+U_att-Vo
        
        #Computes probability using Metropolis Hastings scheme
        minE = -(1/beta)*np.log(N/(rho_res*A*Lambda))
        if E<=minE:
            p =  N*np.exp(beta*E)/(rho_res*A*Lambda)
            return p
        else:
            return 1.0

#Returns positions for bonds such that they immediately bond
#Used for bond angle fluctuation simulations.
def initial_positions_for_bond(params):
    L = params['boxlength']
    N = params['nparticles']
    patch1X = params['p1x']
    patch1Y = params['p2x']
    normP = np.sqrt(patch1X**2+patch1Y**2)
    theta = 20.0*np.pi/180.0
    positions = np.zeros((N,3))
    positions[0,:] = np.array([L/2,L/2,0.0])
    
    for i in range(1,N):
        positions[i,2] = 2.0*theta + positions[i-1,2] 
        positions[i,0] = positions[i-1,0] - 2*normP*np.cos(theta + positions[i-1,2])
        positions[i,1] = positions[i-1,1] - 2*normP*np.sin(theta + positions[i-1,2])
    
    return positions
