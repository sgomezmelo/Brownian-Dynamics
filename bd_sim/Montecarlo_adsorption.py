from funcs import *
from CREATE_PARAMS import *
import numpy as np
import matplotlib.pyplot as plt

TEMP=293.15 #in K
BOLTZMANN_CONSTANT = 1.38e-23  # [k_B] = J/K
V_o = 35*BOLTZMANN_CONSTANT*TEMP  #Attractive potential in units of J
rho = np.linspace(0.1,15, num = 20)*10**23  #Particle concentration of the reservoir
N_eq = np.zeros(len(rho))

def Adsorption(V_o,rho):
    params = create_params(BOXLENGTH = 170e-9,PARTNUM = 20, Vo = V_o, rho = rho, ATT = 0)

    beta = params['beta']
    Vo = params['surface potential']
    L = params['boxlength']

    r = init_particles(params)
    x = r[:,0]
    y = r[:,1]
    dt = 0.05  #Time step in us
    t = 2000  #Total simulation time in us
    N_iterations = int(t/dt)
    t_vector = np.linspace(0,t, num = N_iterations)
    N_particles = np.zeros(N_iterations)
    N_particles[0] = params['nparticles']

    for i in range(1,N_iterations):
        print('e',params['rep_ep'])
        n_current = N_particles[i-1]
        #Attempt particle creation
        x_new = np.random.uniform(0,L)
        y_new = np.random.uniform(0,L)
        
        dx = x-x_new
        dy = y-y_new
        d = np.sqrt(dx**2+dy**2)

        p_create = create_montecarlo(d,0,0,0,0,params,attraction = False)
        draw = np.random.uniform()
        print('Pcreate',p_create)
        if draw<=p_create:
            x = np.concatenate((x,np.array([x_new])))
            y = np.concatenate((y,np.array([y_new])))
            n_current +=1
            
        #Attempt particle destruction
        i_a = np.random.choice(int(n_current))
        x_a = x[i_a]
        y_a = y[i_a]
        
        dx_d = x_a - np.delete(x,i_a)
        dy_d = y_a - np.delete(y,i_a)

        dist_d = np.sqrt(dx_d**2+dy_d**2)
        p_destroy = destroy_montecarlo(dist_d,0,0,0,0,params,attraction = False)
        print('Pdestroy',p_destroy)
        draw = np.random.uniform()
        if draw<=p_destroy:
            x = np.delete(x,i_a)
            y = np.delete(y,i_a)
            n_current -= 1
            
        N_particles[i] = n_current
        print('N',n_current)

    return t_vector, N_particles


for i in range(len(rho)):
    r = rho[i]
    t,N = Adsorption(V_o,r)
    N_eq[i] = N[-1]

 #r = 10**23
 #print(r)
 #t,n = Adsorption(V_o,r)
 #print(n)
plt.figure()
plt.scatter(rho,N_eq)
plt.show()
