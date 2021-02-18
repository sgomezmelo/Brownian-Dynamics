from funcs import *
from forces import *


def total_force_comp(x, y, alpha, p1x, p1y, p2x, p2y, a1x, a1y, a2x, a2y, params):

    '''Step 2 Calculate distances between particles, patches and the angles between patch alignment vectors'''
    # distance, absolute distance
    dist_matrix_x = compute_dist(x, params)
    dist_matrix_y = compute_dist(y, params)
    absolute_dist_matrix = abs_dist_matrix(dist_matrix_x, dist_matrix_y)
   # print('min distance in force',absolute_dist_matrix.min())
   # print(np.unravel_index(np.argmin(absolute_dist_matrix),absolute_dist_matrix.shape ))

    # patch distance, angle 12
    patch_dist_matrix_12_x = compute_dist_patch_AB(p1x, p2x, params)
    patch_dist_matrix_12_y = compute_dist_patch_AB(p1y, p2y, params)
    abs_dist_12 = abs_dist_matrix(patch_dist_matrix_12_x, patch_dist_matrix_12_y)
    theta_12 = compute_angle_patch_AB(a1x, a1y, a2x, a2y)

    # patch distance, angle 21
    patch_dist_matrix_21_x = - patch_dist_matrix_12_x.T
    patch_dist_matrix_21_y =  - patch_dist_matrix_12_y.T
    abs_dist_21 =  abs_dist_12.T
    theta_21 = theta_12.T



    '''Step 3 Calculate repulsive force matrices'''
    f_rep_matrix_x = np.array(rep_force(dist_matrix_x, absolute_dist_matrix, params))
    f_rep_matrix_y = np.array(rep_force(dist_matrix_y, absolute_dist_matrix, params))

    # calculate total repulsive force (array)
    total_f_rep_x = total_force(f_rep_matrix_x)
    total_f_rep_y = total_force(f_rep_matrix_y)

    # turn repulsive force off
    # total_f_rep_x = np.array([0.0]*PARTICLE_NUMBER)
    # total_f_rep_y = np.array([0.0]*PARTICLE_NUMBER)

    '''Step 4 Calculate attractive force matrices'''

    f_att_matrix_x_12 = np.array(attr_force_AB(patch_dist_matrix_12_x, theta_12, abs_dist_12, params))
    f_att_matrix_x_21 = - f_att_matrix_x_12.T

    f_att_matrix_y_12 = np.array(attr_force_AB(patch_dist_matrix_12_y, theta_12, abs_dist_12, params))
    f_att_matrix_y_21 = - f_att_matrix_y_12.T

    # needed to calculate the torque that act on patch A and B
    total_f_att_matrix_x_patch1 = f_att_matrix_x_12
    total_f_att_matrix_x_patch2 = f_att_matrix_x_21

    total_f_att_matrix_y_patch1 = f_att_matrix_y_12
    total_f_att_matrix_y_patch2 = f_att_matrix_y_21

    # calculate total repulsive force on each patch (array)
    total_f_att_x_patch1 = total_force(total_f_att_matrix_x_patch1)
    total_f_att_x_patch2 = total_force(total_f_att_matrix_x_patch2)
    total_f_att_y_patch1 = total_force(total_f_att_matrix_y_patch1)
    total_f_att_y_patch2 = total_force(total_f_att_matrix_y_patch2)

    '''Step 5 Calculate Total Force'''
    total_force_x = total_f_rep_x + total_f_att_x_patch1 + total_f_att_x_patch2
    total_force_y = total_f_rep_y + total_f_att_y_patch1 + total_f_att_y_patch2


    #calculate orientational torque
    orientational_torque_matrix_21 = np.array(orientational_torque_AB(theta_21, abs_dist_21, a2x, a2y, a1x, a1y, params))
    orientational_torque_matrix_12 = np.array(orientational_torque_AB(theta_12, abs_dist_12, a1x, a1y, a2x, a2y, params))

    total_orientational_torque_matrix = orientational_torque_matrix_12 + orientational_torque_matrix_21
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

def total_force_comp_with_tail(x, y, alpha, p1x, p1y, p2x, p2y, a1x, a1y, a2x, a2y, sspx, sspy, params):

    '''Step 2 Calculate distances between particles, patches and the angles between patch alignment vectors'''
    # distance, absolute distance
    dist_matrix_x = compute_dist(x, params)
    dist_matrix_y = compute_dist(y, params)
    absolute_dist_matrix = abs_dist_matrix(dist_matrix_x, dist_matrix_y)

    # distance for the tails
    dist_matrix_tail_x = compute_dist(sspx, params)
    dist_matrix_tail_y = compute_dist(sspy, params)
    absolute_dist_matrix_tail = abs_dist_matrix(dist_matrix_tail_x, dist_matrix_tail_y)

    # distance tail com
    dist_matrix_tail_com_x = compute_dist_tail_com(sspx, x, params)
    dist_matrix_tail_com_y = compute_dist_tail_com(sspy, y, params)
    abs_dist_tail_com = abs_dist_matrix(dist_matrix_tail_com_x, dist_matrix_tail_com_y)

    # distance com tail
    dist_matrix_com_tail_x = -dist_matrix_tail_com_x.T
    dist_matrix_com_tail_y =-dist_matrix_tail_com_y.T
    abs_dist_com_tail = abs_dist_tail_com.T


    # patch distance, angle 12
    patch_dist_matrix_12_x = compute_dist_patch_AB(p1x, p2x, params)
    patch_dist_matrix_12_y = compute_dist_patch_AB(p1y, p2y, params)
    abs_dist_12 = abs_dist_matrix(patch_dist_matrix_12_x, patch_dist_matrix_12_y)
    theta_12 = compute_angle_patch_AB(a1x, a1y, a2x, a2y)

    # patch distance, angle 21
    patch_dist_matrix_21_x = - patch_dist_matrix_12_x.T
    patch_dist_matrix_21_y =  - patch_dist_matrix_12_y.T
    abs_dist_21 =  abs_dist_12.T
    theta_21 = theta_12.T
     #patch_dist_matrix_21_x = compute_dist_patch_AB(p2x, p1x, params)
     #patch_dist_matrix_21_y = compute_dist_patch_AB(p2y, p1y, params)
     #abs_dist_21 = abs_dist_matrix(patch_dist_matrix_21_x, patch_dist_matrix_21_y)
     #theta_21 = compute_angle_patch_AB(a2x, a2y, a1x, a1y)

    '''Step 3 Calculate repulsive force matrices'''
    f_rep_matrix_x = np.array(rep_force(dist_matrix_x, absolute_dist_matrix, params))
    f_rep_matrix_y = np.array(rep_force(dist_matrix_y, absolute_dist_matrix, params))

    f_rep_matrix_tail_x = np.array(rep_force(dist_matrix_tail_x, absolute_dist_matrix_tail, params, tails=True))
    f_rep_matrix_tail_y = np.array(rep_force(dist_matrix_tail_y, absolute_dist_matrix_tail, params, tails=True))

    f_rep_matrix_tail_com_x = np.array(rep_force(dist_matrix_tail_com_x, abs_dist_tail_com, params, tailscom=True))
    f_rep_matrix_tail_com_y = np.array(rep_force(dist_matrix_tail_com_y, abs_dist_tail_com, params, tailscom=True))

    f_rep_matrix_com_tail_x = -f_rep_matrix_tail_com_x.T
    f_rep_matrix_com_tail_y = -f_rep_matrix_tail_com_y.T

    force_on_tail_matrix_x = f_rep_matrix_tail_x + f_rep_matrix_tail_com_x
    force_on_tail_matrix_y = f_rep_matrix_tail_y + f_rep_matrix_tail_com_y

    # calculate total repulsive force (array)
    total_f_rep_x = total_force(f_rep_matrix_x + f_rep_matrix_com_tail_x + force_on_tail_matrix_x)
    total_f_rep_y = total_force(f_rep_matrix_y + f_rep_matrix_com_tail_y + force_on_tail_matrix_y)

    # turn repulsive force off
    # total_f_rep_x = np.array([0.0]*PARTICLE_NUMBER)
    # total_f_rep_y = np.array([0.0]*PARTICLE_NUMBER)

    '''Step 4 Calculate attractive force matrices'''

    f_att_matrix_x_12 = np.array(attr_force_AB(patch_dist_matrix_12_x, theta_12, abs_dist_12, params))
    f_att_matrix_x_21 =  - f_att_matrix_x_12.T

    f_att_matrix_y_12 = np.array(attr_force_AB(patch_dist_matrix_12_y, theta_12, abs_dist_12, params))
    f_att_matrix_y_21 =  - f_att_matrix_y_12.T

    # needed to calculate the torque that act on patch A and B
    total_f_att_matrix_x_patch1 = f_att_matrix_x_12
    total_f_att_matrix_x_patch2 = f_att_matrix_x_21

    total_f_att_matrix_y_patch1 = f_att_matrix_y_12
    total_f_att_matrix_y_patch2 = f_att_matrix_y_21

    # calculate total repulsive force on each patch (array)
    total_f_att_x_patch1 = total_force(total_f_att_matrix_x_patch1)
    total_f_att_x_patch2 = total_force(total_f_att_matrix_x_patch2)
    total_f_att_y_patch1 = total_force(total_f_att_matrix_y_patch1)
    total_f_att_y_patch2 = total_force(total_f_att_matrix_y_patch2)

    total_f_tail_x = total_force(force_on_tail_matrix_x)
    total_f_tail_y = total_force(force_on_tail_matrix_y)

    '''Step 5 Calculate Total Force'''
    total_force_x = total_f_rep_x + total_f_att_x_patch1 + total_f_att_x_patch2
    total_force_y = total_f_rep_y + total_f_att_y_patch1 + total_f_att_y_patch2

    #calculate orientational torque
    orientational_torque_matrix_12 = np.array(orientational_torque_AB(theta_12, abs_dist_12, a1x, a1y, a2x, a2y, params))
    orientational_torque_matrix_21 = np.array(orientational_torque_AB(theta_21, abs_dist_21, a2x, a2y, a1x, a1y, params))
    total_orientational_torque_matrix = orientational_torque_matrix_12 + orientational_torque_matrix_21
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



 #p1 = np.array([1,2,3])
 #p2 =  np.array([4,5,6])
 #params = {'boxlength' : 5}
 #print(compute_dist_patch_AB(p1, p2, params))
 #print(compute_dist_patch_AB(p2, p1, params))
