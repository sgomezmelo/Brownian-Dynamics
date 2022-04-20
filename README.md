# Brownian Dynamics/Grand Canonical simulation of SAS-6 self assembly

## Introduction

The code in this repository implements a coupled Brownian Dynamics/Grand Canonical Monte Carlo (BD/GCMC) routine to model the self assembly of the Spindle Assembly Abnormal Protein 6 (SAS-6) into nonameric rings. The algorithm solves the Langevin equations of motion of suitably coarse grained particles on a 2D fluid in contact with a reservoir which allows for particle exchange, thus modelling adsorption. The repository also includes a file for the fitting of coagulation fragmentation equations (CF) in the reaction limited approximation to the simulated concentrations in time. This is the official repository of -.

## Libraries

The routine employs the folowing libraries:

- Numpy
- NetworkX
- JSON
- Scipy

## Minimal Example
