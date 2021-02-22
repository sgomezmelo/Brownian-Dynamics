import numpy as np
import matplotlib.pyplot as plt
import CREATE_PARAMS
subfolder = 'DataAnalysis_cluster/Results/'
p = np.load(subfolder+'params.npz', allow_pickle=True )
params = p['params'].item()
n_trials = 5 
o = np.zeros((int(params['simsteps']/params['saveint']),20))
c = np.zeros((int(params['simsteps']/params['saveint']),20))
for k in range(n_trials):
    data_o = np.load(subfolder+'open'+str(k)+'.npz', allow_pickle = True)
    data_c = np.load(subfolder+'closed'+str(k)+'.npz', allow_pickle = True)
    o += data_o[data_o.files[0]]/n_trials
    c += data_c[data_c.files[0]]/n_trials


time =  np.load(subfolder+'time0.npz', allow_pickle = True)
t = time[time.files[0]]*params['tscale']*1e+6
plt.figure()
plt.plot(t,o[:,1])
plt.savefig('monomer_simsteps_'+str(params['simsteps'])+'_Montecarlo_steps'+str(params['Montecarlo step'])+'.png')
row, col = o.shape
plt.figure()
for i in range(2,col):
    if i<11:
         plt.plot(t,o[:,i], label =  str(i)+' open')
plt.ylabel('N')
plt.xlabel('t (us)')
plt.legend()
plt.savefig('open_simsteps_'+str(params['simsteps'])+'_Montecarlo_steps'+str(params['Montecarlo step'])+'.png')

plt.figure()
row, col = c.shape
plt.figure()
for i in range(1,col):
    if i>8 and i<13:
        plt.plot(t,c[:,i], label = str(i)+' closed')

plt.legend()
plt.ylabel('N')
plt.xlabel('t (us)')
plt.savefig('closed_simsteps_'+str(params['simsteps'])+'_Montecarlo_steps'+str(params['Montecarlo step'])+'.png')
