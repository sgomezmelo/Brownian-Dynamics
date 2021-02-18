from CALC_TRAJECTORY import calc_trajectory
from CREATE_PARAMS import create_params
from funcs import init_particles
import numpy as np
import matplotlib.pyplot as plt
import os

def bd(x, y, alpha, params, run=5):
    no_open_complexes, no_closed_complexes, t = calc_trajectory(x, y, alpha, run, params)
    print("SIMULATION RUN ", run, ".............SUCCESSFUL")

    return True

n_trials = 5
params = create_params(BOXLENGTH = 170e-9,SIMSTEPS=600000, ONLY_STATISTICS = True, N_trials = n_trials)
N = params['nparticles']
L = params['boxlength']
folder =  params['MAIN_PATH']
subfolder = params['MAIN_PATH'] + params['SUB_PATH']

os.system('mkdir ' + folder)
os.system('mkdir ' + subfolder)
os.system('touch '+ subfolder + "Parameters.txt")
for k in range(n_trials):
    positions = init_particles(params)
    x = positions[:,0]
    y =  positions[:,1]
    alpha = np.random.uniform(0,2*np.pi,N)
    bd(x,y,alpha,params, run = k)



