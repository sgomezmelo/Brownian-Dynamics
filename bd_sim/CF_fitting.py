#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 14:10:40 2021

@author: santiagogomez
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import solve_ivp, odeint
import sys
import proplot as pplt
sys.path.append('/Users/santiagogomez/Downloads/Pplot')
from pubproplot import *

#Initialize variables
N = 10
N_min_closed = 8
N_closed = (N-N_min_closed)
#folder = 'avg_micro_param/'
folder = 'DataAnalysis_Cluster/average_multiple_parameter/'
name = str(sys.argv[1])
MC = int(sys.argv[2])
e_att = float(sys.argv[3])
time = np.loadtxt(folder+name+'_time'+'.csv', delimiter = ',')
max_time = int(len(time))
t = time[:max_time]

#Extract relevant data of Concentrations and time
N_particles = np.loadtxt(folder+name+'_average_total_number'+'.csv', delimiter = ',')
N_particles = N_particles[:max_time]
closedC = np.loadtxt(folder+name+'_average_closed'+'.csv', delimiter = ',')
closedC = closedC[:max_time,N_min_closed:N]
openC= np.loadtxt(folder+name+'_average_open'+'.csv', delimiter = ',')
openC = openC[:max_time, 1:N+1]

#Single array of concentrations
C = np.concatenate((openC,closedC), axis = 1).T
params = np.load(folder+'params.npz', allow_pickle=True)
params = params['params'].item()
dt = params['tstep']
initial_particles = params['nparticles']
k_i = 1.0/(MC*dt)

No_parameters_to_fit = 2*(N_closed)+2
k0 = 7.0*np.ones(No_parameters_to_fit)
print(k0)

#First fit adsorption Kinetics using total particle number
def FD(x,x0,alpha):
    f = np.zeros(x.shape)
    f[x <= x0+15.0/alpha] = 1.0/(1.0+np.exp(alpha*(x-x0)))
    return f

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

def Adsorption_cost_function(x, N_obs, k_i, t,initial_particles):
    k = np.exp(x)
    y0 = [initial_particles]
    sol= solve_ivp(dNdt, (t[0],t[-1]), y0, method = 'BDF' , args = (k,k_i), t_eval = t, atol = 0.1)
    Estimated_N = sol.y
    weights = np.ones(len(N_obs))
    weights[N_obs>0.0] = 1/N_obs[N_obs>0.0]
    #print('Current adsorption iter', x)
    res = np.sum((weights*(Estimated_N-N_obs))**2)/len(N_obs)
    print('Error', res)
    return res
    
r0 = np.ones(2) 
r0[0] = 3.0
ads_result = minimize(Adsorption_cost_function, r0, args = (N_particles, k_i, t,initial_particles), method = 'Nelder-Mead')
r_constants = np.exp(ads_result.x)
print(r_constants)
y0 = [initial_particles]
N_solution = solve_ivp(dNdt, (t[0],t[-1]), y0, method = 'RK45' , args = (r_constants,k_i), t_eval = t)
np.savetxt(folder+name+'_Adsorption_kinetics_solution.csv',N_solution.y,delimiter = ',')

latexify()
fig = pplt.figure(refaspect = 1.61)
ax = fig.add_subplot(111)
ax.scatter(t,N_particles, label = 'GCMC')
ax.plot(t,(N_solution.y)[0,:], label = 'RSA')
ax.set_xlabel(r'$\bar{t}$')

ax.set_ylabel(r'$N_t$')
format_axes(ax)
ax.legend(loc = 'lower right', ncols = 1)
plt.savefig('Adsorption_kinetics.pdf')
print('Finished fitting adsorption parameters')

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




def Cost_finite_dif(x, C_obs_full, N_species, N_min, t_full, k_i, ads_params, time_step = 20 , include_monomers = True):
    C_obs = C_obs_full[:,::time_step]
    t = t_full[::time_step]
    k = np.exp(-x)
    dCobs = C_obs[:,2:]-C_obs[:,:-2]
    dt = t[2:]-t[:-2]
    #print(dt)
    dCdt_obs = dCobs/(dt[0])
    dC_CF = vectorized_CF(C_obs,N_species, N_min, k,k_i, ads_params)
    weights = np.ones(dCdt_obs.shape)
    #weights[dCdt_obs!=0.0] =  1/dCdt_obs[dCdt_obs!=0.0]
    error = np.sum(((dCdt_obs-dC_CF[:,1:-1])/weights)**2)/C_obs.size
    print(error)
    print(x)
    return error

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

result = minimize(CostFunction, k0, args = (C, N, N_min_closed, t, k_i, r_constants), method = 'Nelder-Mead', options={'maxiter':2000})
print(result)
k = np.exp(-result.x)
print(k)
y0 = np.zeros(N+N_closed)
y0[0] = initial_particles
solution = solve_ivp(CF, (0.0,time[-1]), y0, method = 'BDF' , args = (N, N_min_closed, k,k_i, r_constants), t_eval = time)
C_est = solution.y
C_est_open = C_est[:N,:].T
print(C_est_open.shape)
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
print(kinetic_constants)
#np.savetxt(folder+name+'_CF_open_solution.csv',C_est_open,delimiter = ',')
#np.savetxt(folder+name+'_CF_closed_solution.csv',C_est_closed,delimiter = ',')
#np.savez(folder+name+'CF_constants',kinetic_constants)


fig = pplt.figure(refaspect = 1.61)
pplt.rc.cycle = '538'
ax = fig.add_subplot(111)
format_axes(ax)

for i in range(8):
    #ax.plot(t,C_test[i,:])
    color = next(ax._get_lines.prop_cycler)['color']
    ax.plot(t,openC[:,i],color = color, label = str(i+1),alpha = 0.2)
    ax.plot(t,C_est_open[:,i], color = color)
ax.legend(loc = 'r', ncols = 1, frame = False)
ax.set_ylabel('N')
ax.set_xlabel(r'$\bar{t}$')
plt.show()

fig = pplt.figure(refaspect = 1.61)
ax = fig.add_subplot(111)
print(closedC[:,0])
for i in range(N_closed):
    color = next(ax._get_lines.prop_cycler)['color']
    ax.plot(t,closedC[:,i],color = color, label = str(i+N_min_closed), alpha = 0.2)
    ax.plot(t,C_est_closed[:,i], color = color)
ax.set_ylabel('N')
ax.set_xlabel(r'$\bar{t}$')
ax.legend(loc = 'r', ncols = 1, frame = False)
plt.show()
#plt.plot(t,C_est[0,:])
# plt.savefig('closed_CF.pdf')
# plt.figure()
# #plt.plot(t,(N_solution.y)[0,:]-np.sum(sizes[1:,np.newaxis]*C_est[1:,:], axis = 0), '--')
# plt.plot(t,C[0,:], alpha = 0.2)
# plt.plot(t,C_est[0,:])
# plt.plot(t,C_est[1,:])
# plt.plot(t,C[1,:], alpha = 0.5)
# plt.plot(t,C_est[2,:])
# plt.plot(t,C[2,:], alpha = 0.5)
# plt.plot(t,C_est[3,:])
# plt.plot(t,C[3,:], alpha = 0.5)
# plt.plot(t,C_est[4,:])
# plt.plot(t,C[4,:], alpha = 0.5)
# plt.show()
# def vectorized_CF(Concentration, N_max, N_min_closed,k,k_ads,ads_constants,open_system = True):
    
#     N_closed = N_max - N_min_closed
#     #N_closed = 0
#     # k_on = k[0]
#     # k_off = k[1]
#     # k_open = k[2:N_closed+2]
#     # k_close = k[N_closed+2:]
#     #k_on = k[0]
#     k_off = 0.0
#     k_on = k[0]
#     k1 = k[1:]
#     #k_off = k[1]
#     #k_open = k[1:N_closed+1]
#     #k_close = k[N_closed+1:]
#     #Extract the number of species that can form ring
    
#     #Extract open and closed concentrations from an overall Concentration Matrix
#     C = Concentration[:N_max,:]
#     C_closed = Concentration[N_max:N_max+N_closed,:]

#     #Prepare empty arrays with derivatives
#     dCdt = np.zeros(C.shape)
#     dC_closed = np.zeros(C_closed.shape)
    
#     #dCdt[0,:] =  (-2*C[0,:]*k_on*np.sum(C[:-1,:],axis = 0) + 2*k_off*np.sum(C[1:,:],axis = 0))
#     dCdt[0,:] =  (-2*C[0,:]*np.sum(C[:-1,:]*k1[:-1,np.newaxis],axis = 0) + 2*k_off*np.sum(C[1:,:],axis = 0))
#     #dCdt[-1] = (k_on*np.sum(C[:-1]*C[-2::-1]) - k_off*N_max*C[-1])
    
#     for i in range(1,N_max):
#         #Compute the terms of the CF equations except the closure and opening of rings
#         # delta = np.ones(len(C[:N-i-1]))
#         # if len(C[:N-i-1])>i:
#         #     delta[i] = 2.0
            
#         # delta2 = np.ones(len(C[i+1:]))
#         # if len(delta2)>2*i:
#         #     delta2[2*i] = 2.0
            
#         # if i%2 != 0:
#         #     n_conf = i
#         # else:
#         #     n_conf = i-1
        
#         # dCdt[i] = (k_on*np.sum(C[:i]*C[i-1::-1]) - 2*k_on*C[i]*np.sum(delta*C[:N-i-1])
#         # + 2*k_off*np.sum(delta2*C[i+1:])-k_off*n_conf*C[i])
        
#         dCdt[i,:] = (k_on*np.sum(C[1:i,:]*(C[:i-1,:][::-1,:]),axis = 0)+k1[i]*C[0,:]*C[i-1,:] - 2*k_on*C[i,:]*np.sum(C[1:N_max-i-1,:],axis = 0)
#         - 2*k1[i]*C[i,:]*C[0,:] + 2*k_off*np.sum(C[i+1:,:],axis = 0)-k_off*i*C[i,:])
        
        
#         if i>= N_min_closed:
#             #Account for opening and closure of rings
#             #print(i-N_min_closed)
#             #dCdt[i,:] += - k_close[i-N_min_closed]*C[i,:] + k_open[i-N_min_closed]*C_closed[i-N_min_closed,:]
#             #dC_closed[i-N_min_closed,:] = k_close[i-N_min_closed]*C[i,:] - k_open[i-N_min_closed]*C_closed[i-N_min_closed,:]
#             dC_closed[i-N_min_closed] = 0
#     #Add source term from the reservoir
#     size = np.arange(N_max)+1.0
#     #dCdt[0] = dNdt(t,N_tot,ads_constants,k_ads,params) - np.sum(size[1:]*dCdt[1:])- np.sum(dC_closed)
#     if open_system == True:
#         c_size = np.zeros(N_max+N_closed)
#         c_size[:N_max] = np.arange(N_max+1.0)[1:]
#         c_size[N_max:] = np.arange(N_closed)+N_min_closed
#         N_tot = np.sum(c_size[:,np.newaxis]*Concentration[:N_max+N_closed,:],axis = 0)
#         #dCdt[0] += k_ads*FD(N_tot,x0,a)*(1.0) + (1.0-FD(N_tot,x0,a))*k_des*(N_j-N_tot)**3
#         dCdt[0,:] += dNdt(t,N_tot,ads_constants,k_ads)
#     dCondt = np.concatenate((dCdt,dC_closed))
#     return dCondt
#