# Discovering Fingerprint of Functional Connectivity Using DNN (fingerprintFCdnn)

![fig](https://github.com/bsplku/fingerprintFCdnn/blob/main/README_fig.png?raw=true)

Here, we provide a set of codes to build and train deep neural network (DNN) model to identify individuals in HCP 1200 dataset using time-varying functional connectivity (tvFC) patterns.
* Based on Python 3.6
* TensorFlow 1.15
* [Other reauired libraries] numpy, timeit, zipfile, os, datetime, pytz, scipy, functools, matplotlib  
<br/>

## main_indiv_identification.ipynb
The main code is for identifying 10 exemplary subjects using 15 s window tvFC patterns. \
Hidden layers are intialized using a pretained model.

## modules_indiv_identification.py
This script includes modules used in the main code.

## data/
This directory includes the input tvFC data.  
The subdirectory RS1 and RS2 indicate resting-state fMRI scans from first and second visits, respectively.
Each visit had two runs with different phase encoding directions, i.e., right-to-left (RL) and left-to-right (LR).

## results_example/
This directory contains an example result. \
Similarlu, the main code will automatically create './results/results_...\_tgHSP\_...' directory. 

<br/><br/>

### Author
>[Juhyeon Lee](jh0104lee@gmail.com) \
>[Brain Signal Processing Lab](https://bspl-ku.github.io/) \
>Department of Brain and Cognitive Engineering, Korea University, Seoul, Republic of Korea

https://github.com/bsplku/fingerprintFCdnn
