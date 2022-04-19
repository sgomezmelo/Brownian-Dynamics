from funcs import *
from forces import *

def total_force_comp(x, y, alpha, p1x, p1y, p2x, p2y, a1x, a1y, a2x, a2y, params):

    '''Step 2 Calculate distances between particles, patches and the angles between patch alignment vectors'''
    # Calculate dX and dY differences between particle bodies
    dist_matrix_x = compute_dist(x, params) # Matrix of Dx between COMs Xi-Xj
    dist_matrix_y = compute_dist(y, params) # Matrix of Dy distances between COMs Yi-Yj
    absolute_dist_matrix = abs_dist_matrix(dist_matrix_x, dist_matrix_y) #Distances between COMs


    #Patch distance, angle 12
    patch_dist_matrix_12_x = compute_dist_patch_AB(p1x, p2x, params) # Matrix Xipatch1 - Xjpatch2
    patch_dist_matrix_12_y = compute_dist_patch_AB(p1y, p2y, params) # Matrix Xipatch1 - Xjpatch2
    abs_dist_12 = abs_dist_matrix(patch_dist_matrix_12_x, patch_dist_matrix_12_y)
    theta_12 = compute_angle_patch_AB(a1x, a1y, a2x, a2y)
    
    #Distances and angles are reflexive
    abs_dist_21 =  abs_dist_12.T
    theta_21 = theta_12.T


    '''Step 3 Calculate repulsive force matrices'''
    #Calculates NxN matrices with pairwise forces along x and y
    f_rep_matrix_x = np.array(rep_force(dist_matrix_x, absolute_dist_matrix, params))
    f_rep_matrix_y = np.array(rep_force(dist_matrix_y, absolute_dist_matrix, params))

    # Calculate total repulsive force by summing over one of the axes
    total_f_rep_x = total_force(f_rep_matrix_x)
    total_f_rep_y = total_force(f_rep_matrix_y)


    '''Step 4 Calculate attractive force matrices'''
    #Compute forces between patch 1 of particle i and patch 2 of particle j
    f_att_matrix_x_12 = np.array(attr_force_AB(patch_dist_matrix_12_x, theta_12, abs_dist_12, params))
    f_att_matrix_x_21 = - f_att_matrix_x_12.T #Newtons third law - Forces are equal in magnitude and opposite direction
    
    
    f_att_matrix_y_12 = np.array(attr_force_AB(patch_dist_matrix_12_y, theta_12, abs_dist_12, params))
    f_att_matrix_y_21 = - f_att_matrix_y_12.T #Newtons third law - Forces are equal in magnitude and opposite direction


    # Total sum by summing over one of the axes
    total_f_att_x_patch1 = total_force(f_att_matrix_x_12)
    total_f_att_x_patch2 = total_force(f_att_matrix_x_21)
    total_f_att_y_patch1 = total_force(f_att_matrix_y_12)
    total_f_att_y_patch2 = total_force(f_att_matrix_y_21)

    '''Step 5 Calculate Total Force'''
    total_force_x = total_f_rep_x + total_f_att_x_patch1 + total_f_att_x_patch2
    total_force_y = total_f_rep_y + total_f_att_y_patch1 + total_f_att_y_patch2


    #Orientational torque
    orientational_torque_matrix_21 = np.array(orientational_torque_AB(theta_21, abs_dist_21, a2x, a2y, a1x, a1y, params))
    orientational_torque_matrix_12 = np.array(orientational_torque_AB(theta_12, abs_dist_12, a1x, a1y, a2x, a2y, params))
    #Sum torques from each patch
    total_orientational_torque_matrix = orientational_torque_matrix_12 + orientational_torque_matrix_21
    #Summ all torques
    total_orientational_torque = total_force(total_orientational_torque_matrix)


    return (total_force_x,
            total_force_y,
            total_f_att_x_patch1,
            total_f_att_x_patch2,
            total_f_att_y_patch1,
            total_f_att_y_patch2,
            abs_dist_12,
            abs_dist_21,
            theta_12,
            theta_21,
            absolute_dist_matrix,
            total_orientational_torque)

#@profile
def total_force_comp_with_tail(x, y, alpha, p1x, p1y, p2x, p2y, a1x, a1y, a2x, a2y, sspx, sspy, params):

    '''Step 2 Calculate distances between particles, patches and the angles between patch alignment vectors'''
    # COM distances
    # Matrix Xi-Xj, Xi is the x coordinate of i-th com (antisymmetric)
    dist_matrix_x = compute_dist(x, params) 
    # Matrix Yi-Yj, Yi is the y coordinate of i-th com (antisymmetric)
    dist_matrix_y = compute_dist(y, params) 
    #Distances between coms (symmetric)
    absolute_dist_matrix = abs_dist_matrix(dist_matrix_x, dist_matrix_y) 

    # Tail distances
    # Matrix Xti-Xtj, Xti is the x coordinate of i-th tail (antisymmetric)
    dist_matrix_tail_x = compute_dist(sspx, params) 
    # Matrix Yti-Ytj, Yti is the y coordinate of i-th tail (antisymmetric)
    dist_matrix_tail_y = compute_dist(sspy, params) 
    #Distances among tails (symmetric)
    absolute_dist_matrix_tail = abs_dist_matrix(dist_matrix_tail_x, dist_matrix_tail_y) 

    # COM - Tail distances
    
    # Matrix Xti-Xj, Xti is the x coordinate of i-th com  and Xj is the j-th tail x coordinate
    dist_matrix_tail_com_x = compute_dist_tail_com(sspx, x, params) 
    # Matrix Yti-Yj, Xti is the x coordinate of i-th com  and Xj is the j-th tail x coordinate
    dist_matrix_tail_com_y = compute_dist_tail_com(sspy, y, params)
    # Matrix of distances between com and tail
    abs_dist_tail_com = abs_dist_matrix(dist_matrix_tail_com_x, dist_matrix_tail_com_y)


    # Distances and angles between patches
    # Xipatch1-Xjpatch2
    patch_dist_matrix_12_x = compute_dist_patch_AB(p1x, p2x, params)
    # Yipatch1-Yjpatch2
    patch_dist_matrix_12_y = compute_dist_patch_AB(p1y, p2y, params)
    #Distance between patches
    abs_dist_12 = abs_dist_matrix(patch_dist_matrix_12_x, patch_dist_matrix_12_y)
    #Angle between patch1i and  patch2j
    theta_12 = compute_angle_patch_AB(a1x, a1y, a2x, a2y)
    
    #Distances and angles 21 should be the transpose of the 12
    abs_dist_21 =  abs_dist_12.T
    theta_21 = theta_12.T

    '''Step 3 Calculate repulsive force matrices'''
    f_rep_matrix_x = np.array(rep_force(dist_matrix_x, absolute_dist_matrix, params))
    f_rep_matrix_y = np.array(rep_force(dist_matrix_y, absolute_dist_matrix, params))

    f_rep_matrix_tail_x = np.array(rep_force(dist_matrix_tail_x, absolute_dist_matrix_tail, params, tails=True))
    f_rep_matrix_tail_y = np.array(rep_force(dist_matrix_tail_y, absolute_dist_matrix_tail, params, tails=True))

    f_rep_matrix_tail_com_x = np.array(rep_force(dist_matrix_tail_com_x, abs_dist_tail_com, params, tailscom=True))
    f_rep_matrix_tail_com_y = np.array(rep_force(dist_matrix_tail_com_y, abs_dist_tail_com, params, tailscom=True))
    
    #Repulsive forces COM-Tail should be negative of force Tail COM transposed
    f_rep_matrix_com_tail_x = -f_rep_matrix_tail_com_x.T
    f_rep_matrix_com_tail_y = -f_rep_matrix_tail_com_y.T
    force_on_tail_matrix_x = f_rep_matrix_tail_x + f_rep_matrix_tail_com_x
    force_on_tail_matrix_y = f_rep_matrix_tail_y + f_rep_matrix_tail_com_y

    # Calculate total repulsive force (array)
    total_f_rep_x = total_force(f_rep_matrix_x + f_rep_matrix_com_tail_x + force_on_tail_matrix_x)
    total_f_rep_y = total_force(f_rep_matrix_y + f_rep_matrix_com_tail_y + force_on_tail_matrix_y)

    '''Step 4 Calculate attractive force matrices'''

    f_att_matrix_x_12 = np.array(attr_force_AB(patch_dist_matrix_12_x, theta_12, abs_dist_12, params))
    f_att_matrix_x_21 =  - f_att_matrix_x_12.T
    f_att_matrix_y_12 = np.array(attr_force_AB(patch_dist_matrix_12_y, theta_12, abs_dist_12, params))
    f_att_matrix_y_21 =  - f_att_matrix_y_12.T


    # calculate total repulsive force on each patch (array)
    total_f_att_x_patch1 = total_force(f_att_matrix_x_12)
    #print('patch1 x', total_f_att_x_patch1)
    total_f_att_x_patch2 = total_force(f_att_matrix_x_21)
    #print('patch2 x', total_f_att_x_patch2)
    total_f_att_y_patch1 = total_force(f_att_matrix_y_12)
    #print('patch1 y', total_f_att_y_patch1)
    total_f_att_y_patch2 = total_force(f_att_matrix_y_21)
    #print('patch2 y', total_f_att_y_patch2)

    total_f_tail_x = total_force(force_on_tail_matrix_x)
    total_f_tail_y = total_force(force_on_tail_matrix_y)

    '''Step 5 Calculate Total Force'''
    total_force_x = total_f_rep_x + total_f_att_x_patch1 + total_f_att_x_patch2
    total_force_y = total_f_rep_y + total_f_att_y_patch1 + total_f_att_y_patch2

    #Calculate orientational torque
    orientational_torque_matrix_12 = np.array(orientational_torque_AB(theta_12, abs_dist_12, a1x, a1y, a2x, a2y, params))
    orientational_torque_matrix_21 = np.array(orientational_torque_AB(theta_21, abs_dist_21, a2x, a2y, a1x, a1y, params))
    total_orientational_torque_matrix = orientational_torque_matrix_12 + orientational_torque_matrix_21
    
    #Sum all pairwise torques
    total_orientational_torque = total_force(total_orientational_torque_matrix)

    return (total_force_x,
            total_force_y,
            total_f_att_x_patch1,
            total_f_att_x_patch2,
            total_f_att_y_patch1,
            total_f_att_y_patch2,
            abs_dist_12,
            abs_dist_21,
            theta_12,
            theta_21,
            absolute_dist_matrix,
            total_f_tail_x,
            total_f_tail_y,
            total_orientational_torque)



