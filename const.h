#pragma once
 
static const double PI          = 3.141592658979;

// parameter for generative model
static const double K_CONST     = 400;
static const double G_CONST     = 10;
static const double SIG_O       = G_CONST/2;
static const double SIG_R       = 0.5;
static const double B_LAT       = 1.7;
static const double B_LIN       = B_LAT / 20;
static const double DELTA_T     = 0.004; 

// parameter for UKF
static const double ALPHA_PARAM   = 0.01;
static const double BETA_PARAM    = 2;
