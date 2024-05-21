#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import argparse
from controllers.LQG import LQG
from controllers.WDRC import WDRC
from controllers.DRCE import DRCE
from joblib import Parallel, delayed

import os
import pickle
def save_data(path, data):
    output = open(path, 'wb')
    pickle.dump(data, output)
    output.close()
    
def main():
    # change True to False if you don't want to use given lambda
    use_lambda = False
    use_optimal_lambda = True
    lambda_ = 20 # will not be used if the parameter "use_lambda = False"
    
    
    #theta_v_list = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0] # radius of noise ambiguity set
    theta_w_list = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    theta_x0 = 0.5
    num_noise_list = [10, 15, 20] 
    noisedist = ['normal']
    for noise_dist in noisedist:
        for theta_w in theta_w_list:
                WDRC_lambda, DRCE_lambda = [],[]
                theta = theta_w
                #for theta in theta_v_list:
                for idx, num_noise in enumerate(num_noise_list):
                    # Use N = N_w = N_v = N_x0
                    num_samples = num_noise
                    num_x0_samples = num_noise
                    theta_ = f"_{str(theta).replace('.', '_')}" # change 1.0 to 1_0 for file name
                    
                    # Load Lambda
                    WDRC_lambda_file = open('./inputs/OutofSample/N={}/wdrc_lambda_{}.pkl'.format(num_noise,theta_), 'rb')
                    WDRC_lambda = pickle.load(WDRC_lambda_file)
                    WDRC_lambda_file.close()
                    # Load Lambda
                    DRCE_lambda_file = open('./inputs/OutofSample/N={}/drce_lambda_{}and{}.pkl'.format(num_noise,theta_, theta_), 'rb')
                    DRCE_lambda = pickle.load(DRCE_lambda_file)
                    DRCE_lambda_file.close()
                    
                    print("N=",num_samples," theta=",theta_w)
                    print(np.asarray(WDRC_lambda).shape)
                    print(np.asarray(DRCE_lambda).shape)
                    
                    WDRC_lambda_save = WDRC_lambda[-100:]
                    DRCE_lambda_save = DRCE_lambda[-100:]
                    
                    
                    save_data( './inputs/OS/N={}/'.format(num_noise)+ 'wdrc_lambda_' + theta_ + '.pkl', WDRC_lambda_save)
                    save_data( './inputs/OS/N={}/'.format(num_noise)+ 'drce_lambda_' + theta_ + '.pkl', DRCE_lambda_save)
                    





if __name__ == "__main__":
    
    
    main()