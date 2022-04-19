import numpy as np
from numpy import linalg as LA
from numpy import inf
import timeit

# Calculates a NxN matrix of the the force component F, where Fij is the attractive force between the A patch of i-th particle 
# and the B patch of the j-th particle along a particular direction.
def attr_force_AB(dist_AB, theta_AB, abs_dist_AB, params):
    
    #Extract relevant parameters
    epsilon_att = params['att_ep']
    sigma = params['sigma']
    cutoff_att = params['att_c']
    cutoff_angle = params['ang_c']
    dimension = dist_AB.shape[0]
    
    #Initialize force matrix as NxN container
    force = np.zeros([dimension, dimension], dtype=float)
    
    
    #Only patches within distance cutoff_att and angular orientation cutoff att experience non zero force
    Non_zero_force = (abs_dist_AB != 0.0) & (theta_AB<=cutoff_angle)
    Non_zero_force = np.logical_and(Non_zero_force, abs_dist_AB<=cutoff_att)
    
    #Calculate non zero entries
    d = dist_AB[Non_zero_force]
    norm_d = abs_dist_AB[Non_zero_force]
    theta = theta_AB[Non_zero_force]
    unit_vector = 4*epsilon_att*d/norm_d
    rep = 12 * (sigma**12) / (norm_d + (2**(1 / 6)) * sigma)**13
    att = - 6 * sigma**6 / (norm_d + (2**(1 / 6)) * sigma)**7
    s = 0.5 * (np.cos(theta * np.pi / cutoff_angle) + 1)
    force[Non_zero_force] = unit_vector*s*(rep+att)
    return force

#Calculates the contribution on the z torque arising from the angular dependence of the potential
def orientational_torque_AB(theta_AB, abs_dist_AB, align_patch_A_x,align_patch_A_y,align_patch_B_x,align_patch_B_y, params):
    #Extract relevant parameters
    epsilon_att = params['att_ep']
    sigma = params['sigma']
    cutoff_att = params['att_c']
    cutoff_angle = params['ang_c']
    rc = params['rc']

    #Define NxN array to contain torques
    dimension = abs_dist_AB.shape[0]
    torque = np.zeros([dimension, dimension], dtype=float)

    #Compute which entries are non zeri depending on angle and distance constraints
    mask =  np.logical_and(abs_dist_AB<=cutoff_att, theta_AB<=cutoff_angle)

    #Compute non zero torques 
    d = abs_dist_AB[mask]
    theta = theta_AB[mask]
    prefactor = 2*epsilon_att*(np.pi/cutoff_angle)
    potential = ((sigma**12) / (d + (2**(1 / 6)) * sigma)**12 - (sigma**6) / (d + (2**(1 / 6)) * sigma)**6 - (sigma/rc)**12 + (sigma/rc)**6)
    cross_product = align_patch_A_x[:,np.newaxis]*align_patch_B_y - align_patch_A_y[:,np.newaxis]*align_patch_B_x
    torque[mask] = prefactor*potential*np.sin(theta*np.pi/cutoff_angle)*np.sign(cross_product[mask])
    return torque

#Calculate repulsive force along a certain direction
def rep_force(dist, abs_dist, params, tails=False, tailscom=False):
    
    #Extract relevant parameters based on whether the force is tail-tail, body-body or tail-body
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
    
    #Determine which pairs of particles are within interacting distance
    dimension = dist.shape[0]
    mask = abs_dist < cutoff_rep
    d = dist[mask]
    #Compute non zero force
    norm_d = abs_dist[mask]
    force = np.zeros([dimension, dimension], dtype=float)
    prefactor = d * 4 * epsilon_rep
    rep_part = 12 * (sigma**12) / norm_d**14
    att_part = - 6 * (sigma**6) / norm_d**8
    force[mask] = prefactor * (rep_part + att_part)

    return force



    
