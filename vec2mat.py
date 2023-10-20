#!/usr/bin/env python3
"""
@author: Juhyeon Lee [jh0104lee@gmail.com]
        Brain Signal Processing Lab [https://bspl-ku.github.io/]
        Department of Brain and Cognitive Engineering, Korea University, Seoul, Republic of Korea
"""
################################## To transform 1D vectors of FC (input tvFC or fpFC) to 2D matrices. ##################################

import numpy as np

# If the number of ROIs is D, the shape of 2d matrix is (D, D) and the shape of 1d vector is (D*(D-1)/2)
# vecs: (D*(D-1)/2) or (1, D*(D-1)/2) or (n, D*(D-1)/2)
# For the ROI label of D=360, please see Glasser MF, Coalson TS, Robinson EC, Hacker CD, Harwell J, Yacoub E, Ugurbil K, Andersson J, Beckmann CF, Jenkinson M, Smith SM, Van Essen DC (2016): A multi-modal parcellation of human cerebral cortex. Nature 536:171–178.
def vec2mat(vecs, D=360):
    upper_tri_idx = np.triu(np.ones((D,D), dtype=int), k=1)

    if np.ndim(vecs)==1:
        vecs = np.reshape(vecs, (1,-1))

    n = np.size(vecs,0)
    matrices = np.zeros((n,D,D))
    for i in range(n):
        vec = vecs[i,:]

        # If vector dimension is 62835, we excluded five ROIs among 360
        if np.size(vec)==((D-5)*(D-5-1)/2):
            excluded_ROI = np.array([110, 120, 290, 292, 300], dtype=np.int16())-1

            ids_mat = np.zeros((D,D), dtype=int)
            ids_mat[upper_tri_idx==1] = np.arange(D*(D-1)/2)

            # Find which edges are excluded
            select_mat = np.ones((D,D), dtype=int)
            for roi in excluded_ROI:
                select_mat[roi,:]=0
                select_mat[:,roi]=0
            select_mat = np.triu(select_mat, k=1)
            select_edges = ids_mat[select_mat==1]

            # Make it 64620 =(D*(D-1)/2)
            vec_old = vec[:]
            vec = np.zeros(((int(D*(D-1)/2))))
            vec[select_edges] = vec_old

        # Fill up the matrix
        matrix = np.zeros((D,D))
        matrix[upper_tri_idx==1] = vec

        matrices[i,:,:] = matrix

    return matrices
