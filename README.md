# Brownian Dynamics/Grand Canonical simulation of SAS-6 self assembly

## Introduction

The code in this repository implements a coupled Brownian Dynamics/Grand Canonical Monte Carlo (BD/GCMC) routine to model the self assembly of the Spindle Assembly Abnormal Protein 6 (SAS-6) into nonameric rings. The algorithm solves the Langevin equations of motion of suitably coarse grained particles on a 2D fluid in contact with a reservoir which allows for particle exchange, thus modelling adsorption via statistical simulation of GCMC. The repository also includes a file for the fitting of coagulation fragmentation equations (CF) in the reaction limited approximation to the simulated concentrations in time. This is the official repository of -.

## Libraries

The routine employs the folowing libraries:

- Python 3.0 or later
- Numpy
- NetworkX
- JSON
- Scipy

## Getting Started

The code runs according to the parameters specified on the "args.txt" file. Here the user can specify the number of time steps to compute, the name of the files on which the results are written, or the name of the folder in which they are saved. Users are welcome to modify these parameters as they see fit. To compute a single run of the code, simply run the "BD_WRAPPER.py" python script:

```
python3 BD_WRAPPER.py
```

Relevant parameters such as the strength or anisotropy of the interactions, attractive potential of the adsorbent surface, length of the simulation domain, or initial number of particles may also be specified in the "args.txt" file. Other parameters, such as the diffusivity of SAS-6 in the particle fixed frame, the temperature or the length scales of the potentials may be specified in the "CREATE_PARAMS.py" script. Keep in mind that the code is designed for a diagonal diffusion matrix. 

## Minimal Example
