# Towards Robust Federated Analytics via Differentially Private Measurements of Statistical Heterogeneity: Experimental Evaluation

This repository contains all the supporting files for the **experimental section** of the paper **_Towards Robust Federated Analytics via Differentially Private Measurements of Statistical Heterogeneity_**, including all the Python files necessary for anyone to repeat all of the experiments given as evidence for the results in the paper.

## Environment

- Install the [**latest version of Python 3**](https://www.python.org/downloads/), and install the additional packages **idx2numpy, matplotlib, mpmat and numpy** using **PIP**.
- Download the Python file **agm_balle_all_datasets.py**.
- Download the archives *cifar-10-python.tar.gz* and *cifar-100-python.tar.gz* using the instructions from [**Alex Krizhevsky's home page**](https://www.cs.toronto.edu/~kriz/cifar.html), and keep the files **data_batch_x** (each ~30GB, where x ranges from 1 to 5), and the file **train** (~150GB) respectively.
- Download the archives *train-images-idx3-ubyte.gz* and *train-labels-idx1-ubyte.gz* using the instructions from [**Zalando**](https://github.com/zalandoresearch/fashion-mnist), and keep the files **train-images-idx3-ubyte** (~45GB) and **train-labels-idx1-ubyte** (~60KB) respectively.

## Instructions

- After setting up the Python environment and downloading the required files as outlined above, open and run **agm_balle_collective_combined**. There will be a progress bar and various text updates appearing in the terminal, indicating the progress of the experiments.
- When the final text update "Finished." appears, the experiments have finished, the final tables have been printed in the terminal, and all plots have been saved in the same folder as the Python files. This should happen after approximately **6-12 hours**, depending on the computer or laptop used.
- These final plots should be **exactly the tables and graphs found in the experimental section** of **_Towards Robust Federated Analytics via Differentially Private Measurements of Statistical Heterogeneity_**.

## Authors

- **[Mary Scott](https://mary-python.github.io/)**, Department of Computer Science, University of Warwick
- **[Graham Cormode](http://dimacs.rutgers.edu/~graham/)**, Department of Computer Science, University of Warwick
- **[Carsten Maple](https://warwick.ac.uk/fac/sci/wmg/people/profile/?wmgid=1102)**, WMG, University of Warwick
