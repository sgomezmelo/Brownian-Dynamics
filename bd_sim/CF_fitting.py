#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 14:10:40 2021

@author: santiagogomez
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import json

#Extract names and arguments from txt file
with open('args_CF_fit.txt','r') as f:
    data = f.read()
print(data)
args = json.loads(data)

#Extract files from simulation and relevant parameters, based on txt file
N = int(args['N max'])
N_min_closed = int(args['N min'])
name = str(args['Final File name'])
folder = args['Results Folder']
params = np.load(folder+args['Parameters'], allow_pickle=True)
params = params['params'].item()
MC = int(params['Montecarlo step'])
dt = params['tstep']
t = np.loadtxt(folder+args['Time file'], delimiter = ',')
N_particles = np.loadtxt(folder+args['Total Number file'], delimiter = ',')
closedC = np.loadtxt(folder+args['Closed concentrations file'], delimiter = ',')
openC= np.loadtxt(folder+args['Open concentrations file'], delimiter = ',')

#Extract relevant concentrations and parameters according to predefined Nmin and Nmax
N_closed = (N-N_min_closed)
closedC = closedC[:,N_min_closed:N]
openC = openC[:, 1:N+1]
C = np.concatenate((openC,closedC), axis = 1).T #Single array of concentrations
initial_particles = params['nparticles']
k_i = 1.0/(MC*dt)
No_parameters_to_fit = 2*(N_closed)+2
k0 = 7.0*np.ones(No_parameters_to_fit)
r0 = np.ones(2) 
r0[0] = 3.0

print(k0)

#Define adsorption kinetics as a second order polynomial, according to RSA kinetics in low limit
def dNdt(t,N,rate_constants,k_i):
    L = params['boxlength']
    R1 = params['RBIG']
    R2 = params['RSMALL']
    maxN = L**2/(np.pi*(R1**2+R2**2))
    k1 = rate_constants[0]
    k2 = rate_constants[1]
    x = N/maxN
    dN = k_i-k1*x+k2*x**2
    return dN

#Define cost function for adsorption kinetics as the sum of squared errors
def Adsorption_cost_function(x, N_obs, k_i, t,initial_particles):
    k = np.exp(x)
    y0 = [initial_particles]
    sol= solve_ivp(dNdt, (t[0],t[-1]), y0, method = 'BDF' , args = (k,k_i), t_eval = t, atol = 0.1)
    Estimated_N = sol.y
    weights = np.ones(len(N_obs))
    weights[N_obs>0.0] = 1/N_obs[N_obs>0.0]
    print('Current adsorption iter', x)
    res = np.sum((weights*(Estimated_N-N_obs))**2)/len(N_obs)
    return res

#Calculates the derivative dCdt according to the simplified, discrete coagulation fragmentation equations
# k - Rate constants vector, C - Concentration of open species, C_closed-concentration of rings, N_min_closed - lowest possible ring
def CF(t,Concentration, N_max, N_min_closed,k,k_ads,ads_constants,open_system = True):
    #Extract the number of species that can form ring
    N_closed = N_max - N_min_closed
    k_on = k[0]
    k_off = k[1]
    k_open = k[1:N_closed+1]
    k_close = k[N_closed+1:]
    #Extract open and closed concentrations from an overall Concentration Matrix
    C = Concentration[:N_max]
    C_closed = Concentration[N_max:N_max+N_closed]
    #Prepare empty arrays with derivatives
    dCdt = np.zeros(N_max)
    dC_closed = np.zeros(len(C_closed))
    
    #Compute the first derivative of the monomers
    dCdt[0] =  (-2*k_on*C[0]*np.sum(C[:-1]) + 2*k_off*np.sum(C[1:]))
    
    for i in range(1,N_max):
        #Compute the terms of the CF equations except the closure and opening of rings
        dCdt[i] = (k_on*np.sum(C[:i]*(C[:i][::-1])) - 2*k_on*C[i]*np.sum(C[:N_max-i-1])
        + 2*k_off*np.sum(C[i+1:])-k_off*i*C[i])
        
        if i>= N_min_closed:
            #Account for opening and closure of rings
            dCdt[i] += - k_close[i-N_min_closed]*C[i] + k_open[i-N_min_closed]*C_closed[i-N_min_closed]
            dC_closed[i-N_min_closed] = k_close[i-N_min_closed]*C[i] - k_open[i-N_min_closed]*C_closed[i-N_min_closed]
    #Add source term from the reservoir
    if open_system == True:
        #c_size = np.arange(N_max+1.0)[1:]
        c_size = np.zeros(N_max+N_closed)
        c_size[:N_max] = np.arange(N_max+1.0)[1:]
        c_size[N_max:] = np.arange(N_closed)+N_min_closed
        N_tot = np.sum(c_size*Concentration[:N_max+N_closed])
        dCdt[0] += dNdt(t,N_tot,ads_constants,k_ads)
    dCondt = np.concatenate((dCdt,dC_closed))
    return dCondt

#Cost function for CF equations
def CostFunction(x, C_obs, N_species, N_min, t, k_i, ads_params , initial_monomers = 1, include_monomers = True):
    
    X = np.exp(-x)
    N_closed = N_species-N_min
    y0 = np.zeros(N_species+N_closed)
    y0[0] = initial_monomers
    
    #Propagate CF equations in time with stiff-suitable solver
    solution = solve_ivp(CF, (0.0,t[-1]), y0, method = 'Radau' , args = (N_species, N_min, X,k_i, ads_params), t_eval = t, atol = 1.0)
    Estimated_c = solution.y
    weights = np.ones(C_obs.shape)
    weights[C_obs>0.0] = 1.0/(C_obs[C_obs>0.0])
    if include_monomers:
        residuals = np.sum(weights*((Estimated_c-C_obs))**2)/C_obs.size
        
    else: 
        residuals = np.sum(weights[1:,:]*((Estimated_c[1:,:]-C_obs[1:,:]))**2)/C_obs.size
    print('current iter',x)
    print('residuals', residuals)
    return residuals

#First fit adsorption parameters
ads_result = minimize(Adsorption_cost_function, r0, args = (N_particles, k_i, t,initial_particles), method = 'Nelder-Mead')
r_constants = np.exp(ads_result.x)
y0 = [initial_particles]
N_solution = solve_ivp(dNdt, (t[0],t[-1]), y0, method = 'RK45' , args = (r_constants,k_i), t_eval = t)
np.savetxt(folder+name+'_adsorption_kinetics_solution.csv',N_solution.y,delimiter = ',')
print('Finished fitting adsorption parameters')

result = minimize(CostFunction, k0, args = (C, N, N_min_closed, t, k_i, r_constants), method = 'Nelder-Mead', options={'maxiter':1000})
k = np.exp(-result.x)
y0 = np.zeros(N+N_closed)
y0[0] = initial_particles
solution = solve_ivp(CF, (0.0,t[-1]), y0, method = 'BDF' , args = (N, N_min_closed, k,k_i, r_constants), t_eval = t)
print(solution)
C_est = solution.y
C_est_open = C_est[:N,:].T
C_est_closed = C_est[N:N+N_closed+1,:].T

kinetic_constants = {
    'N': N,
    'N_min_closed':N_min_closed,
    'N_closed': N_closed,
    'k_on':k[0],
    'k_off':k[1],
    'k_open':k[1:N_closed+1],
    'k_close':k[N_closed+1:2*N_closed+1]
    }
np.savetxt(folder+name+'_open_solution.csv',C_est_open,delimiter = ',')
np.savetxt(folder+name+'_closed_solution.csv',C_est_closed,delimiter = ',')
np.savez(folder+name+'_rate_constants',kinetic_constants)

print('Finished')

