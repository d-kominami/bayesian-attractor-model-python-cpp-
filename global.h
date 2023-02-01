#pragma once
#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <fstream>
#include <random>
#include <cmath>
#include "Eigen/Core"
#include "Eigen/Geometry"
#include "Eigen/Cholesky"
#include "Eigen/Dense"
#include "const.h"
using namespace std;
using namespace Eigen;

random_device seed_gen;
default_random_engine engine(1);

MatrixXd sigmoid(MatrixXd);
MatrixXd generative_model(MatrixXd);

double   prob_mnorm(MatrixXd, MatrixXd, MatrixXd);
bool     DEBUG_MSG_ON;
double   UNCTY_Q; 
double   UNCTY_R;

int      NUM_METRICS   = 1;
int      NUM_STATE     = 1;
int      SIGMA_POINTS  = 3;
int      L_PARAM       = NUM_STATE;
double   KAPPA_PARAM   = 3 - static_cast<double>(L_PARAM);
double   LAMBDA_PARAM  = ALPHA_PARAM*ALPHA_PARAM *(L_PARAM+KAPPA_PARAM)-L_PARAM;
double   GAMMA_PARAM   = sqrt(L_PARAM+LAMBDA_PARAM);

int      UKF(MatrixXd);
void     init_z(void);
void     change_f_dim(int);
void     change_k_dim(int);
void     init_attractor(void);
void     init_nrm_variables(void);
void     set_d_uncertain(double);
void     set_s_uncertain(double);
void     upd_feature(MatrixXd);
void     debug_msg(int);
void     set_norm_param(MatrixXd,MatrixXd);

MatrixXd system_average     = MatrixXd::Zero(NUM_STATE, 1);
MatrixXd system_covariance  = MatrixXd::Identity(NUM_STATE, NUM_STATE);
MatrixXd system_noise_w     = MatrixXd::Zero(NUM_STATE, NUM_STATE);
MatrixXd system_noise_v     = MatrixXd::Zero(NUM_METRICS, NUM_METRICS);
MatrixXd feature_vector     = MatrixXd::Zero(NUM_METRICS, NUM_STATE);
MatrixXd attractor_vector   = MatrixXd::Zero(NUM_STATE, NUM_STATE);
MatrixXd confidence         = MatrixXd::Zero(NUM_STATE, 1);
MatrixXd normalized_m       = MatrixXd::Zero(NUM_METRICS, 1);
MatrixXd normalized_s       = MatrixXd::Zero(NUM_METRICS, 1);
bool is_normalize_param_set = false;

normal_distribution<> dist(0.0, sqrt(5)); // z 初期化

class bam{
public:
    bam(int,int,double,double);
    MatrixXd get_z(void);
    MatrixXd get_p(void);
    MatrixXd get_c(void);
    void ukf_z(MatrixXd);
    void set_f_dim(int);
    void set_k_dim(int);
    void set_q(double);
    void set_r(double);
    void msg_on(void);
    void upd_f(MatrixXd);
    void set_norm_prm(MatrixXd,MatrixXd);
};
