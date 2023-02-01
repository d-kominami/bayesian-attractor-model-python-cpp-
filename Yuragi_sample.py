import bam_module
import numpy as np
import sys

if __name__ == '__main__':

    # create an instance of Bayesian attractor model
    # arguments
    ### number of attractors
    ### number of dimension of a input feature 
    ### dynamics uncertainty
    ### sensory uncertainty
    #
    # sample
    # python3 Yuragi_sample.py sample_feature.txt sample_input.txt
    #
    BAM = bam_module.bam(3, 3, 2, 1)

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
           



