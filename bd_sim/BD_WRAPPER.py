from CALC_TRAJECTORY import calc_trajectory
from CREATE_PARAMS import create_params
import numpy as np
import sys
import os
from funcs import *
import json

#Read input arguments from text file
with open('args.txt','r') as f:
    data = f.read()
args = json.loads(data)

n_trials = 1
repulsive_energy = args['Repulsive Energy Strength'] #Repulsive energy parameter in k_BT
index = int(args['Index'])
file_name = str(args['Name'])
atractive_energy = args['Attractive Energy Strength']
L = float(args['Boxlength']) #Length of the box side in m
N_steps = int(args['Number of time steps'])
initial_N = int(args['Initial Number of Particles']) #Initial number of particles
N_MC = int(args['GCMC time step']) #Number of steps to attempt montecarlo
V_surface = float(args['Surface Potential'])  #surface potential in kbT
angle = float(args['Cutoff angle'])
subpath = str(args['Subpath'])

#@profile
def bd(x, y, alpha, params, file_name, run=5):
    no_open_complexes, no_closed_complexes, t = calc_trajectory(x, y, alpha, run, file_name, params)
    print("SIMULATION RUN ", run, ".............SUCCESSFUL")

    return True


params = create_params(ANG = angle, BOXLENGTH = L, SIMSTEPS = N_steps, ONLY_STATISTICS=True,  
                       GRAPHICAL = True , N_trials = n_trials, TAIL = True, REP = repulsive_energy, 
                       TREP = repulsive_energy, ATT = atractive_energy, Vo = V_surface,
                       montecarlo_step = N_MC, PARTNUM = initial_N, open_system = True, SUB_PATH = subpath,
                       include_attraction_GCMC = False, surface_exclusion = False)
print(params['att_c'])
N = params['nparticles']
L = params['boxlength']
folder =  params['MAIN_PATH']
subfolder = params['MAIN_PATH'] + params['SUB_PATH']
print('START')
os.system('mkdir ' + folder)
os.system('mkdir ' + subfolder)
os.system('touch '+ subfolder + "Parameters.txt")
print('END')

positions = init_particles(params)
x = positions[:,0]
y =  positions[:,1]
alpha = np.random.uniform(0,2*np.pi,N)
bd(x,y,alpha,params, file_name , run = index)



