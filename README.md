# Towards Robust Federated Analytics via Differentially Private Measurements of Statistical Heterogeneity: Experimental Evaluation

This repository contains all the supporting files for the **experimental section** of the paper **_Towards Robust Federated Analytics via Differentially Private Measurements of Statistical Heterogeneity_**, including all the Python files necessary for anyone to repeat all of the experiments given as evidence for the results in the paper.

## Environment

- Install the [**latest version of Python 3**](https://www.python.org/downloads/), and install the additional packages **idx2numpy, matplotlib, mpmath and numpy** using **PIP**.
- Download the Python file **agm_balle_all_datasets.py**.
- Download the archives *cifar-10-python.tar.gz* and *cifar-100-python.tar.gz* using the instructions from [**Alex Krizhevsky's home page**](https://www.cs.toronto.edu/~kriz/cifar.html), and keep the files **data_batch_x** (each ~30GB, where x ranges from 1 to 5), and the file **train** (~150GB) respectively.
- Download the archives *train-images-idx3-ubyte.gz* and *train-labels-idx1-ubyte.gz* using the instructions from [**Zalando**](https://github.com/zalandoresearch/fashion-mnist), and keep the files **train-images-idx3-ubyte** (~45GB) and **train-labels-idx1-ubyte** (~60KB) respectively.

## Instructions

- After setting up the Python environment and downloading the required files as outlined above, open and run **agm_balle_collective_combined**.
- After each dataset has been analyzed, the tables are printed in the terminal and the plots are saved in the same folder as the Python files. This should happen after approximately 30 minutes.
- These tables and graphs should be **exactly those found in the experimental section** of **_Towards Robust Federated Analytics via Differentially Private Measurements of Statistical Heterogeneity_**.

## Authors

- **[Mary Scott](https://mary-python.github.io/)**, Department of Computer Science, University of Warwick
- **[Graham Cormode](http://dimacs.rutgers.edu/~graham/)**, Department of Computer Science, University of Warwick
- **[Carsten Maple](https://warwick.ac.uk/fac/sci/wmg/people/profile/?wmgid=1102)**, WMG, University of Warwick
