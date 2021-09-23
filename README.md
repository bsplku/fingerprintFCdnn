# fingerprintFCdnn

![alt text](https://https://github.com/bsplku/fingerprintFCdnn/edit/main/README_fig.png?raw=true)

Here, we provide a set of codes to build and train deep neural network (DNN) model to identify individuals in HCP 1200 dataset using time-varying functional connectivity (tvFC) patterns.
* Based on Python 3.6
* TensorFlow 1.15
* [Other reauired libraries] numpy, timeit, zipfile, os, datetime, pytz, scipy, functools, matplotlib

## main_indiv_identification.ipynb
The main code is for identifying 10 exemplary subjects using 15 s window tvFC patterns. \
Hidden layers are intialized using a pretained model.

## modules_indiv_identification.py
This script includes modules used in the main code.

## data/
This directory includes the input data.

## results_example/
This directory contains example result. \
The main you will automatically create './results/results_..._tgHSP_...' directory which is similar to this. 
