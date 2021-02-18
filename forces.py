import numpy as np
from numpy import linalg as LA
from numpy import inf
import timeit

def attr_force_AB(dist_AB, theta_AB, abs_dist_AB, params):
    epsilon_att = params['att_ep']
    sigma = params['sigma']
    cutoff_att = params['att_c']
    cutoff_angle = params['ang_c']
    dimension = dist_AB.shape[0]
    force = np.zeros([dimension, dimension], dtype=float)

    Non_zero_force = np.logical_and(abs_dist_AB<=cutoff_att, theta_AB<=cutoff_angle)
    Non_zero_force = np.logical_and(Non_zero_force, abs_dist_AB>1e-10)
    d = dist_AB[Non_zero_force]
    norm_d = abs_dist_AB[Non_zero_force]
    theta = theta_AB[Non_zero_force]
    unit_vector = 4*epsilon_att*d/norm_d
    rep = 12 * (sigma**12) / (norm_d + (2**(1 / 6)) * sigma)**13
    att = - 6 * sigma**6 / (norm_d + (2**(1 / 6)) * sigma)**7
    s = 0.5 * (np.cos(theta * np.pi / cutoff_angle) + 1)
    force[Non_zero_force] = unit_vector*s*(rep+att)
    return force


  #def attr_force_AB(dist_AB, theta_AB, abs_dist_AB, params):


     # epsilon_att = params['att_ep']
      #sigma = params['sigma']
      #cutoff_att = params['att_c']
      #cutoff_angle = params['ang_c']

      #dimension = dist_AB.shape[0]
      #force = np.zeros([dimension, dimension], dtype=float)
      #mask_dist = (abs_dist_AB < cutoff_att).astype(int)
      #mask_theta = (theta_AB < cutoff_angle).astype(int)
      #mask = ((mask_dist + mask_theta) == 2).astype(int)
      #index_list = np.argwhere(mask == 1)

      #for k in range(len(index_list)):
         # idx_i = index_list[k,0]
          #idx_j = index_list[k,1]
          #if idx_i != idx_j:
            #  if abs_dist_AB[idx_i, idx_j] < 1e-10:
              #    force[idx_i, idx_j] = 0
              #else:
                #  prefactor = dist_AB[idx_i, idx_j] * 4 * (epsilon_att / abs_dist_AB[idx_i, idx_j])
                  #rep_part = 12 * (sigma**12) / (abs_dist_AB[idx_i, idx_j] + (2**(1 / 6)) * sigma)**13
                  #att_part = - 6 * sigma**6 / (abs_dist_AB[idx_i, idx_j] + (2**(1 / 6)) * sigma)**7
                  #switch = 0.5 * (np.cos(theta_AB[idx_i, idx_j] * np.pi / cutoff_angle) + 1)
                  #force[idx_i, idx_j] = prefactor * (rep_part + att_part) * switch


    # dimension = dist_AB.shape[0]
    # zero = np.zeros([dimension, dimension], dtype=float)
    #
    # mask_dist = abs_dist_AB >= cutoff_att
    # mask_theta = theta_AB >= cutoff_angle
    # prefactor = dist_AB * 4 * (epsilon_att / abs_dist_AB)
    # prefactor[prefactor == inf] = 0.
    #
    # rep_part = 12 * (sigma**12) / (abs_dist_AB + (2**(1 / 6)) * sigma)**13
    # np.fill_diagonal(rep_part, 0.0)
    #
    # att_part = - 6 * sigma**6 / (abs_dist_AB + (2**(1 / 6)) * sigma)**7
    # np.fill_diagonal(att_part, 0.0)
    #
    # switch = 0.5 * (np.cos(theta_AB * np.pi / cutoff_angle) + 1)
    #
    # force = prefactor * (rep_part + att_part) * switch
    #
    # np.copyto(force, zero, casting='same_kind', where=mask_dist)
    # np.copyto(force, zero, casting='same_kind', where=mask_theta)

    return force

#def orientational_torque_AB(theta_AB, abs_dist_AB, align_patch_A_x,
 #                                                  align_patch_A_y,
  #                                                 align_patch_B_x,
   #                                                align_patch_B_y, params):


    # print("align_patch_A_x", align_patch_A_x)
    # print("align_patch_A_y", align_patch_A_y)
    # print("align_patch_B_x", align_patch_B_x)
    # print("align_patch_B_y", align_patch_B_y)
    #epsilon_att = params['att_ep']
    #sigma = params['sigma']
    #cutoff_att = params['att_c']
    #cutoff_angle = params['ang_c']
    #rc = params['rc']

    #dimension = abs_dist_AB.shape[0]
    #torque = np.zeros([dimension, dimension], dtype=float)
    #mask_dist = (abs_dist_AB < cutoff_att).astype(int)
    #mask_theta = (theta_AB < cutoff_angle).astype(int)
    #mask = ((mask_dist + mask_theta) == 2).astype(int)
    #index_list = np.argwhere(mask == 1)
    # print("mask", mask)
    # print(index_list)
    #for k in range(len(index_list)):
     #   idx_i = index_list[k,0]
      #  idx_j = index_list[k,1]
       # if idx_i != idx_j:
            # print("idx_i", idx_i)
            # print("idx_j", idx_j)
        #    prefactor = 2*epsilon_att*(np.pi/cutoff_angle)

         #   if theta_AB[idx_i, idx_j] < 1e-5:
          #      angular_part = np.pi/cutoff_angle
           # else:
            #    angular_part = np.sin(theta_AB[idx_i, idx_j] * np.pi / cutoff_angle)/np.sqrt(1-(np.cos(theta_AB[idx_i, idx_j]))**2)

            #potential = ((sigma**12) / (abs_dist_AB[idx_i, idx_j] + (2**(1 / 6)) * sigma)**12
             #            - (sigma**6) / (abs_dist_AB[idx_i, idx_j] + (2**(1 / 6)) * sigma)**6
              #           - (sigma/rc)**12
               #          + (sigma/rc)**6)
            #crossproduct = (align_patch_A_x[idx_i]*align_patch_B_y[idx_j]
             #               -align_patch_A_y[idx_i]*align_patch_B_x[idx_j])#np.sin(theta_AB[idx_i, idx_j])
            #torque[idx_i, idx_j] = prefactor * angular_part * potential * crossproduct
            # print("theta21 in force = ", theta_AB)
            # print("abstand in force = ", abs_dist_AB)
            # print("angular_part", angular_part)
            # print("potential", potential)
            # print("crossproduct =", crossproduct)
            # print("torque", torque)
    #return torque

def orientational_torque_AB(theta_AB, abs_dist_AB, align_patch_A_x,align_patch_A_y,align_patch_B_x,align_patch_B_y, params):
    epsilon_att = params['att_ep']
    sigma = params['sigma']
    cutoff_att = params['att_c']
    cutoff_angle = params['ang_c']
    rc = params['rc']

    dimension = abs_dist_AB.shape[0]
    torque = np.zeros([dimension, dimension], dtype=float)
    mask =  np.logical_and(abs_dist_AB<=cutoff_att, theta_AB<=cutoff_angle)

    d = abs_dist_AB[mask]
    theta = theta_AB[mask]

    A_x =  align_patch_A_x
    A_y =  align_patch_A_y

    B_x =  align_patch_B_x
    B_y =  align_patch_B_y

    prefactor = 2*epsilon_att*(np.pi/cutoff_angle)
    potential = ((sigma**12) / (d + (2**(1 / 6)) * sigma)**12 - (sigma**6) / (d + (2**(1 / 6)) * sigma)**6 - (sigma/rc)**12 + (sigma/rc)**6)

    ang = np.zeros(theta.size)
    ang[theta>1e-5] = np.sin(theta[theta>1e-5] * np.pi / cutoff_angle)/np.sqrt(1-(np.cos(theta[theta>1e-5])**2))
    ang[theta<=1e-5] = np.pi/cutoff_angle

    cross_product = A_x[:,np.newaxis]*B_y - A_y[:,np.newaxis]*B_x
    cross_product = cross_product[mask]

    torque[mask] = prefactor*potential*ang*cross_product

    return torque


def rep_force(dist, abs_dist, params, tails=False, tailscom=False):

     #  print('minimum distance force', dist.min())

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

    dimension = dist.shape[0]
    mask = abs_dist < cutoff_rep
    d = dist[mask]
    norm_d = abs_dist[mask]
    force = np.zeros([dimension, dimension], dtype=float)

    prefactor = d * 4 * epsilon_rep
    rep_part = 12 * (sigma**12) / norm_d**14
    att_part = - 6 * (sigma**6) / norm_d**8

    force[mask] = prefactor * (rep_part + att_part)

    return force


 #def rep_force(dist, abs_dist, params, tails=False, tailscom=False):

   #  if tails is True:
     #    epsilon_rep = params['tail_rep_ep']
       #  sigma = params['sigma_tails']
         #cutoff_rep = params['rep_tails']

     #if tailscom is True:

       #  epsilon_rep = params['tail_rep_ep']
        # sigma = params['sigma_tail_com']
         #cutoff_rep = params['rep_tail_com']

     #if (tails is False and tailscom is False):

       #  epsilon_rep = params['rep_ep']
         #sigma = params['sigma']
         #cutoff_rep = params['rep_c']

     #dimension = dist.shape[0]
     #mask = (abs_dist < cutoff_rep).astype(int)
     #index_list = np.argwhere(mask == 1)
     #force = np.zeros([dimension, dimension], dtype=float)

     #for k in range(len(index_list)):
       #  idx_i = index_list[k,0]
         #idx_j = index_list[k,1]
         #if idx_i != idx_j:
           #  prefactor = dist[idx_i, idx_j] * 4 * epsilon_rep
             #rep_part = 12 * (sigma**12) / abs_dist[idx_i, idx_j]**14
             #att_part = - 6 * (sigma**6) / abs_dist[idx_i, idx_j]**8

             #force[idx_i, idx_j] = prefactor * (rep_part + att_part)

    # dimension = dist.shape[0]
    # zero = np.zeros([dimension, dimension], dtype=float)
    # mask = abs_dist >= cutoff_rep

    # prefactor = dist * 4 * epsilon_rep
    # rep_part = 12 * (sigma**12) / abs_dist**14
    # att_part = - 6 * (sigma**6) / abs_dist**8
    # rep_part[rep_part == inf] = 0
    # att_part[att_part == -inf] = 0
    # force = prefactor * (rep_part + att_part)
    # np.fill_diagonal(force, 0.0)
    # np.copyto(force, zero, casting='same_kind', where=mask)

     #return force


#params = {'att_ep': 1.0, 'sigma' : 1, 'att_c' : 100, 'ang_c': 1.0, 'rc': 1}
#d = np.array([[0,1],[1,0]])
#theta = np.array([[0.5,0.1],[0.05,0.5]])*np.pi
#A_x =  np.array([1,0.5])
#A_y =  np.array([2,0])

#B_x =  np.array([1,1])
#B_y =  np.array([0,-3])

#print(orientational_torque_AB(theta, d, A_x,A_y,B_x,B_y, params))
#print(fast_torque(theta, d, A_x,A_y,B_x,B_y, params))




#test1 = '''
#N = 100
#d = np.random.uniform(size = (N,N))
#theta =  np.random.uniform(size = (N,N))*np.pi
#A_x = np.random.uniform(size = (N))
#A_y = np.random.uniform(size = (N))
#B_x = np.random.uniform(size = (N))
#B_y = np.random.uniform(size = (N))
#params = {'att_ep': 1.0, 'sigma' : 1, 'att_c' : 100, 'ang_c': 1.0, 'rc': 1}


#fast_torque(theta, d, A_x,A_y,B_x,B_y, params)

#'''

#test2 =  '''
#N = 100
#d = np.random.uniform(size = (N,N))
#theta =  np.random.uniform(size = (N,N))*np.pi
#A_x = np.random.uniform(size = (N))
#A_y = np.random.uniform(size = (N))
#B_x = np.random.uniform(size = (N))
#B_y = np.random.uniform(size = (N))
#params = {'att_ep': 1.0, 'sigma' : 1, 'att_c' : 100, 'ang_c': 1.0, 'rc': 1}

#orientational_torque_AB(theta, d, A_x,A_y,B_x,B_y, params)

#'''


#print(fast_dist(d,theta,abs_d,params))
#print(attr_force_AB(d,theta,abs_d,params))

#print(timeit.timeit(test1,'from __main__ import fast_torque , np', number = 100))
#print(timeit.timeit(test2,'from __main__ import orientational_torque_AB , np', number = 100))

    
