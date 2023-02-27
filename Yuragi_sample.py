import bam_module
import numpy as np
import sys

if __name__ == '__main__':

    # create an instance of Bayesian attractor model
    # 4 arguments are:
    ### number of attractors
    ### number of dimension of a input feature 
    ### dynamics uncertainty
    ### sensory uncertainty
    #
    # sample
    # python3 Yuragi_sample.py sample_feature.txt sample_input.txt
    #
    BAM = bam_module.bam(3, 3, 2, 1)
    # Note: this version does not support multiple instance creation.
    # if you try to create another instance, 
    ### (BAM2 = bam_module.bam(3, 12, 1, 2))
    # Variables BAM and BAM2 refer to the same memory space. 

    feature = np.loadtxt(sys.argv[1])
    input   = np.loadtxt(sys.argv[2])
    normprm = np.loadtxt(sys.argv[3])

    BAM.set_norm_prm(normprm[0],normprm[1])
    BAM.upd_f(feature)
    #BAM.msg_on()
    
    with open('res.csv', mode='a') as f:

        for row in range(input.shape[0]):
            observation = input[row].reshape(1,input.shape[1])
            BAM.ukf_z(observation)
            # z  = BAM.get_z()
            c  = BAM.get_c()
            for row in c:
                print (row[0], ",", file=f, end=" ")
            print("", file=f)
           



