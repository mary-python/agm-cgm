# Towards Robust Federated Analytics via Differentially Private Measurements of Statistical Heterogeneity: Experimental Evaluation

This repository contains all the supporting files for the **experimental section** of the paper **_Towards Robust Federated Analytics via Differentially Private Measurements of Statistical Heterogeneity_**, including all the Python files necessary for anyone to repeat all of the experiments given as evidence for the results in the paper.

## Environment

- Install the [**latest version of Python 3**](https://www.python.org/downloads/), and install the additional packages **idx2numpy, matplotlib, mpmath, numpy and PIL** using **PIP**.
- Download the Python file **agm_balle_collective_combined.py**.
- Download the archive *cifar-10-python.tar.gz* using the instructions from [**Alex Krizhevsky's home page**](https://www.cs.toronto.edu/~kriz/cifar.html), and keep the files **data_batch_x** (each ~30GB), where x ranges from 1 to 5.
- Download the archive *cifar-100-python.tar.gz* using the instructions from [**Alex Krizhevsky's home page**](https://www.cs.toronto.edu/~kriz/cifar.html), and keep the file **train** (~150GB).
- Download the file **train-images-idx3-ubyte** (7657 KB) from [**Zalando**](https://github.com/zalandoresearch/fashion-mnist).
- Download the folder *dataset_dir* using the instructions from [**Apple**](https://github.com/apple/ml-flair), and keep the subfolder *small_images* (1.11 GB) only. There is no need to download the raw full-size images.

## Instructions

- After setting up the Python environment and downloading the required files as outlined above, open and run **agm_balle_collective_combined**. There will be a progress bar and various text updates appearing in the terminal, indicating the progress of the experiments.
- When the final text update "Finished." appears, the experiments have finished and the final plots have been saved in the same folder as the Python files. This should happen after approximately **6-12 hours**, depending on the computer or laptop used.
- These final plots should be **exactly the graphs found in the experimental section** of **_Towards Robust Federated Analytics via Differentially Private Measurements of Statistical Heterogeneity_**.

## Authors

- **[Mary Scott](https://mary-python.github.io/)**, Department of Computer Science, University of Warwick
- **[Graham Cormode](http://dimacs.rutgers.edu/~graham/)**, Department of Computer Science, University of Warwick
- **[Carsten Maple](https://warwick.ac.uk/fac/sci/wmg/people/profile/?wmgid=1102)**, WMG, University of Warwick
