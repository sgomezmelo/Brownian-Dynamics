# Grand canonical Brownian dynamics simulations of adsorption and
self-assembly of SAS-6 rings on a surface

## Introduction

The code in this repository implements a coupled Brownian Dynamics/Grand Canonical Monte Carlo (BD/GCMC) routine to model the self-assembly of the Spindle Assembly Abnormal Protein 6 (SAS-6) into nonameric rings. The algorithm solves the Langevin equations of motion of suitably coarse grained particles on a 2D fluid in contact with a reservoir which allows for particle exchange, thus modelling adsorption via statistical simulation of GCMC. The repository also includes a file for the fitting of coagulation fragmentation equations (CF) in the reaction limited approximation to the simulated concentrations in time. This is the official repository of the manuscript "Grand canonical Brownian dynamics simulations of adsorption and
self-assembly of SAS-6 rings on a surface".

## Libraries

The routine employs the folowing libraries:

- Python 3.0 or later
- Numpy
- NetworkX
- JSON
- Scipy

## Setup

The code runs according to the parameters specified on the "args.txt" file. Here the user can specify the number of time steps to compute, the name of the files on which the results are written, or the name of the folder in which they are saved. Users are welcome to modify these parameters as they see fit. To compute a single run of the code, simply run the "BD_WRAPPER.py" python script:

```
python3 BD_WRAPPER.py
```
The results are saved on several files in the specified folder with the specified name, according to the "args.txt" file. The number of protein clusters with different protein numbers at several times are stored in the "closed" file in the case of rings and in the "open" file otherwise; each column represents a cluster size and each row represents a time. The total number of proteins is saved in "total_particle_number" file. The times in which the measurements are taken are stored in the "time" file. The files "part_pos_and_ori" and "patch_pos" track the coordinates of the particles and binding sites in time, respectively. The file "spheres" stores the position of the bead model in the global frame.

## Citation

If you found this code useful, please consider citing. 
