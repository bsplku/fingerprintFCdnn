# fingerprintFCdnn
Here, we provide a set of codes to build and train deep neural network (DNN) model to identify individuals in HCP 1200 dataset using time-varying functional connectivity (tvFC) patterns.
* Python 3.6
* TensorFlow 1.15

## main_indiv_identification.ipynb
The main code is for to identifying 10 exemplary subjects using 15 s window tvFC patterns.
Hidden layers are intialized using a pretained model saved under ./pretrained.

## modules_indiv_identification.py
This script includes modules used in the main code.

## data/
This directory includes the input data.

## pretrained/
This directory includes the pretained model of identifying 300 subjects.
