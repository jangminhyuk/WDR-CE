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
    
    
    # theta_v_list = [0.1, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0] #[1.0, 2.0, 3.0, 4.0, 5.0, 6.0] # radius of noise ambiguity set
    # theta_w_list = [0.1, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    theta_w_list = [0.05, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    num_noise_list = [10, 15, 20] 
    noisedist = ['normal']
    WDRC_lambda = np.zeros((7,7))
    DRCE_lambda = np.zeros((7,7))
    for idx_w, theta_w in enumerate(theta_w_list):
        for num_noise in num_noise_list:
            #theta_v_ = f"_{str(theta).replace('.', '_')}" # change 1.0 to 1_0 for file name
            theta_w_ = f"_{str(theta_w).replace('.', '_')}" # change 1.0 to 1_0 for file name
            # Load Lambda
            WDRC_lambda_file = open('./results/OS/N={}/wdrc_lambda_{}.pkl'.format(num_noise,theta_w_), 'rb')
            WDRC_lambda_ = pickle.load(WDRC_lambda_file)
            WDRC_lambda_file.close()
            # Load Lambda
            DRCE_lambda_file = open('./results/OS/N={}/drce_lambda_{}and{}.pkl'.format(num_noise,theta_w_,theta_w_), 'rb')
            DRCE_lambda_ = pickle.load(DRCE_lambda_file)
            DRCE_lambda_file.close()
            
            print("theta : ",theta_w," N : ", num_noise)
            print(np.array(DRCE_lambda_).shape)
            print(np.array(WDRC_lambda_).shape)
            
            DRCE_lambda_save = DRCE_lambda_[-100:]
            save_data('./inputs/OutOfSample/N={}/DRCE_lambda_{}.pkl'.format(num_noise,theta_w_),DRCE_lambda_save)
            
            WDRC_lambda_save = WDRC_lambda_[-100:]
            save_data('./inputs/OutOfSample/N={}/WDRC_lambda_{}.pkl'.format(num_noise,theta_w_),WDRC_lambda_save)
            
            #WDRC_lambda[idx_w][idx_v] = WDRC_lambda_
            #DRCE_lambda[idx_w][idx_v] = DRCE_lambda_
            
            # # Load Lambda
            # WDRC_lambda_file = open('./results/QQ_longT_tmp/longT_wdrc_lambda_{}and{}.pkl'.format(idx_w,idx_v), 'rb')
            # WDRC_lambda_ = pickle.load(WDRC_lambda_file)
            # WDRC_lambda_file.close()
            # # Load Lambda
            # DRCE_lambda_file = open('./results/QQ_longT_tmp/longT_drce_lambda_{}and{}.pkl'.format(idx_w,idx_v), 'rb')
            # DRCE_lambda_ = pickle.load(DRCE_lambda_file)
            # DRCE_lambda_file.close()
            
            
            # WDRC_lambda[idx_w][idx_v] = WDRC_lambda_
            # DRCE_lambda[idx_w][idx_v] = DRCE_lambda_
                
                
                    
    #save_data('./results/QQ_longT_tmp/longT_WDRC_lambda.pkl',WDRC_lambda)
    #save_data('./results/QQ_longT_tmp/longT_DRCE_lambda.pkl',DRCE_lambda)
    
    print(WDRC_lambda)
    print(DRCE_lambda)




if __name__ == "__main__":
    
    
    main()