# Discovering Fingerprint of Functional Connectivity Using DNN (fingerprintFCdnn)

![fig](https://github.com/bsplku/fingerprintFCdnn/blob/main/README_fig.png?raw=true)

Here, we provide a set of codes to build and train a weight-sparsity-controlled deep neural network (DNN) model to identify individuals in the HCP 1200 dataset using time-varying functional connectivity (tvFC) patterns during a resting state. 

The codes were implemented in the Python 3.6 environment with the following libraries:

* TensorFlow 1.15
* numpy, timeit, zipfile, os, datetime, pytz, scipy, functools, matplotlib  
<br/>

## main_indiv_identification.ipynb
This main code is for identifying 10 example subjects using 15-s window tvFC patterns. \
Hidden layers are initialized using a pretrained model to shorten a training time.

## modules_indiv_identification.py
This module contains the classes/functions used in the main code.

## data/
This directory includes the input tvFC data. \
The subdirectories RS1 (or Day1) and RS2 (or Day2) indicate resting-state fMRI (rfMRI) sessions from the first and second visits, respectively. In each visit, two rfMRI runs were acquired with different phase encoding directions, i.e., right-to-left (RL) and left-to-right (LR).

## results_example/
This directory contains example results. \
Similarly, the main code will store the corresponding results in an automatically created folder under the './results' directory. 

<br/><br/>

### Author
>[Juhyeon Lee](jh0104lee@gmail.com) \
>[Brain Signal Processing Lab](https://bspl-ku.github.io/) \
>[Department of Brain and Cognitive Engineering](https://bce.korea.ac.kr), [Korea University](https://www.korea.edu), Seoul, Republic of Korea

https://github.com/bsplku/fingerprintFCdnn
